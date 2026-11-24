"""Frozen GW-only training and evaluation protocol.

optimize() receives only graph layers and K. Checkpoint and seed selection use
the profiled GW objective; ground-truth labels are used only by evaluate().
"""
import argparse
import copy
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import model_core as R
from features import build_features
from community_template import CommunitySpecificGW
from volume_objective import VolumeRegularizedGW
from mass_scaled_objective import MassScaledGW
from degree_corrected_objective import DegreeCorrectedGW
from dual_community_objective import DualCommunityGW
from profiled_template import template_profile
from coverage_fusion import support_fusion
from data_io import ProfiledGW, load_case, propagation_operator, DT

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = 'V8-dual-GW-four-candidate-discrete-validated-20261005'
SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
VARIANTS={'inverse','community-square','community-kl'}
RESTORE_DEGREE=True

def normalize(H):
    return (H-H.mean(1, keepdim=True))/H.std(1, keepdim=True).clamp_min(1e-8)

class Encoder(R.MultilayerGWEncoder):
    def __init__(self, operators, X, K, steps, multiscale=False, zero_extra=False, hidden_norm=False, early=False, uniform=False):
        super().__init__(operators, X, K, steps)
        self.multiscale=multiscale
        self.hidden_norm=hidden_norm
        self.early=early
        if early:
            base=self.fusion
            replacement=torch.nn.Linear(24,24,dtype=DT)
            with torch.no_grad():
                replacement.weight.copy_(base.weight.reshape(24,len(operators),24).sum(1))
                replacement.bias.copy_(base.bias)
            self.fusion=replacement
            self.layer_logits=torch.nn.Parameter(torch.zeros(len(operators),dtype=DT),requires_grad=not uniform)
        if multiscale:
            # Preserve the exact random baseline parameters, and add separately
            # seeded columns so stochastic construction does not change W/head.
            base=self.fusion
            with torch.random.fork_rng():
                torch.manual_seed(17591)
                extended=torch.nn.Linear(2*base.in_features+24,24,dtype=DT)
            with torch.no_grad():
                extended.weight[:,:base.in_features].copy_(base.weight)
                extended.bias.copy_(base.bias)
                if zero_extra:extended.weight[:,base.in_features:].zero_()
            self.fusion=extended

    def forward(self):
        Z=self.X@self.W0
        operators=self.operators
        if self.early:
            alpha=torch.softmax(self.layer_logits,0)
            operators=(alpha[:,None,None]*operators).sum(0,keepdim=True)
        operators1=operators2=operators
        stabilize=lambda H:(H-H.mean(1,keepdim=True))/((H-H.mean(1,keepdim=True)).square().mean(1,keepdim=True)+1e-6).sqrt()
        first=torch.stack([torch.sparse.mm(op,Z) for op in operators1])
        H1=torch.tanh(stabilize(first) if self.hidden_norm else first)
        second=torch.stack([torch.sparse.mm(op,h) for op,h in zip(operators2,H1@self.W1)])
        H2=torch.tanh(stabilize(second) if self.hidden_norm else second)
        self.spreads={'h1':H1.detach().std(1).mean(1).tolist(),
                      'h2':H2.detach().std(1).mean(1).tolist()}
        flatten=lambda H: H.permute(1,0,2).reshape(len(self.X),-1)
        parts=[flatten(H2 if self.hidden_norm else normalize(H2))]
        if self.multiscale:
            H0=torch.tanh(Z)[None,:,:]
            parts += [flatten(H1 if self.hidden_norm else normalize(H1)), H0[0] if self.hidden_norm else normalize(H0)[0]]
        fused=self.fusion(torch.cat(parts,dim=1))
        self.saturation=float((fused.detach().abs()>3).double().mean())
        return self.head(torch.tanh(fused))

def wide_features(layers,P,seed,K,width=None):
    # width=16 is the explicitly disabled old degree-input control.
    if width is not None:assert width==16
    X=build_features(layers,seed,K,'degree').clone()
    aggregate=P[0]
    for op in P[1:]:aggregate=aggregate+op
    aggregate=(aggregate/len(P)).coalesce()
    channels=X[:,len(layers):]
    for _ in range(8):channels=torch.sparse.mm(aggregate,channels)
    centered=channels-channels.mean(0,keepdim=True)
    X[:,len(layers):]=.25*centered/centered.std(0,keepdim=True).clamp_min(1e-8)
    if width is not None or not RESTORE_DEGREE:X[:,:len(layers)].zero_()
    return X

def matched_encoder(P,X,K,steps):
    # Same constructor/random stream; zero formerly unused input-weight rows.
    net=Encoder(P,X,K,steps,False,False,False,False,False)
    with torch.no_grad():net.W0[:len(P)].zero_()
    return net


def stats(obj,T):
    if isinstance(obj,DualCommunityGW):
        with torch.no_grad():
            W=T/obj.p[:,None];node,degree=obj.layer_parts(T)
            convert=lambda profile:{key:value.detach().tolist() for key,value in profile.items()}
            return dict(q=T.sum(0).tolist(),node_layer_objectives=node.tolist(),degree_layer_objectives=degree.tolist(),node_profile=convert(obj.measure_profile(T,'node')[1]),degree_profile=convert(obj.measure_profile(T,'degree')[1]),total_objective=float(obj.loss(T)),structure_mass_aux=[float(v) for v in obj.parts(T)],entropy=float(-(W*W.log()).sum(1).mean()),row_error=float((T.sum(1)-obj.p).abs().max()),finite=bool(torch.isfinite(T).all()),min_transport=float(T.min()))
    with torch.no_grad():
        if isinstance(obj,(MassScaledGW,DegreeCorrectedGW)):
            target_q,b,a,x=obj.coefficients(T);M=T/obj.p[:,None]
            _,_,gain,_,_=template_profile(target_q,obj.mu,x)
            return {'q':T.sum(0).tolist(),'template_target_masses':target_q.tolist(),
                    'template_statistic':x.tolist(),'theta_out':b.tolist(),'alpha':a.tolist(),
                    'degree_layer_objectives':obj.layer_parts(T)[1].tolist(),'node_layer_objectives':obj.layer_parts(T)[0].tolist(),
                    'total_objective':float(obj.loss(T)),
                    'structure_mass_aux':[float(v) for v in obj.parts(T)],'aux_coefficient_registered':obj.coefficient,
                    'entropy':float(-(M*M.clamp_min(1e-300).log()).sum(1).mean()),
                    'row_error':float((T.sum(1)-obj.p).abs().max()),
                    'finite':bool(torch.isfinite(T).all()),'min_transport':float(T.min()),
                    'q_guard_triggered':bool((target_q<1e-12).any())}
        q=T.sum(0);h=q.square().sum()
        x=obj._x(T)
        z=x-h*obj.mu
        D=1-torch.relu(z).square()/(h*(1-h)*obj.null)
        M=T/obj.p[:,None]
        return {'q':q.tolist(),'total_objective':float(obj.loss(T)),
                    'structure_mass_aux':[float(v) for v in obj.parts(T)],'aux_coefficient_registered':obj.coefficient,'excess':z.tolist(),
                'positive_layers':int((z>0).sum()),
                'entropy':float(-(M*M.clamp_min(1e-300).log()).sum(1).mean()),
                'row_error':float((T.sum(1)-obj.p).abs().max()),
                'finite':bool(torch.isfinite(T).all()),'min_transport':float(T.min())}

def infer(obj,logits,log_eta,damping=1.,smooth=False,create_graph=True):
    if damping != 1. or smooth:
        raise ValueError('Frozen runner supports only the exact KL mirror baseline')
    return R.unroll(obj,logits,log_eta,create_graph)

def optimize(layers,K,seed,epochs,steps,variant,threads=1,identity=False,lr=.012,schedule='constant',aux_weight=.25,optimizer='adam',checkpoint=False,spectral=None,starting_state=None,starting_provenance=None,input_seed=None):
    # No truth labels, masks, ARI, or external community assignments exist here.
    started=time.perf_counter(); R.seed_all(seed); torch.set_num_threads(threads)
    if variant not in VARIANTS:raise ValueError(f'Unknown variant: {variant}')
    if steps<0 or epochs<1:raise ValueError('steps must be nonnegative and epochs positive')
    obj=DegreeCorrectedGW(layers,K,coefficient=.5) if variant=='inverse' else DualCommunityGW(layers,K,cost='square' if variant=='community-square' else 'kl')
    obj.bounded_step=True
    obj.row_bounded=True
    if 'volume' in variant:obj.weight=.05
    if 'node' in variant:obj.q_floor=1/(4*layers[0].shape[0])
    P=[propagation_operator(A) for A in layers]
    if identity:raise ValueError('The sparse verification runner implements only the frozen baseline')
    feature_seed=seed if input_seed is None else input_seed
    X=wide_features(layers,P,feature_seed,K)
    net=matched_encoder(P,X,K,steps)
    smooth=variant=='continuous'
    if smooth:
        with torch.no_grad():net.log_eta.fill_(np.log(.1/.9))
    if starting_state is not None:
        if starting_provenance is None:raise ValueError('Starting provenance required')
        net.load_state_dict(starting_state)
    elif starting_provenance is not None:raise ValueError('Unexpected starting provenance')
    initial_state=copy.deepcopy(net.state_dict())
    opt=torch.optim.Adam(net.parameters(),lr=lr)
    aux=aux_weight if 'aux' in variant else 0.
    damping=.5 if 'damped' in variant else 1.
    random_logits=net()
    random_encoder=R.initial_transport(obj,random_logits).detach()
    random_final,_=infer(obj,random_logits.detach(),net.log_eta,damping,smooth,False)
    random_final=random_final.detach()
    trace=[];loss_path=[]; encoder_path=[]; zero_epochs=0;checks=[]
    best_loss=float('inf');best_state=None;best_epoch=None
    for ep in range(epochs):
        if schedule=='cosine':
            factor=.01+.99*(1+np.cos(np.pi*ep/max(epochs-1,1)))/2
            opt.param_groups[0]['lr']=lr*factor
        opt.zero_grad(); logits=net()
        encoded=R.initial_transport(obj,logits)
        final,history=infer(obj,logits,net.log_eta,damping,smooth)
        encoded_loss=obj.loss(encoded); final_loss=obj.loss(final)
        loss=final_loss+aux*encoded_loss
        if checkpoint and float(final_loss.detach())<best_loss:
            best_loss=float(final_loss.detach());best_state=copy.deepcopy(net.state_dict());best_epoch=ep
        if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite loss')
        loss.backward()
        norm=sum(float(p.grad.square().sum()) for p in net.parameters() if p.grad is not None)**.5
        zero_epochs+=int(norm==0)
        loss_path.append(float(final_loss.detach()));encoder_path.append(float(encoded_loss.detach()))
        if ep in {0,9,49,99,199,299,599,epochs-1}:
            trace.append({'epoch':ep,'final_loss':loss_path[-1], 'encoded_loss':encoder_path[-1],
                          'total_loss':float(loss.detach()),'gradient_norm':norm,'lr':opt.param_groups[0]['lr'],
                          'degree_input_gradient_norm':float(net.W0.grad[:len(layers)].norm()),'random_input_gradient_norm':float(net.W0.grad[len(layers):].norm()),
                          'gradients':{n:float(p.grad.norm()) if p.grad is not None else None for n,p in net.named_parameters()},
                          'spreads':net.spreads,'fusion_saturation':net.saturation,
                          'history':history,'encoded_stats':stats(obj,encoded), 'final_stats':stats(obj,final)})
        if optimizer=='checked' and steps:raise ValueError('Checked training is currently specified for continuous encoder, S=0 only')
        params=[p for p in net.parameters() if p.requires_grad and p.grad is not None]
        before=[p.detach().clone() for p in params] if optimizer=='checked' else None
        gradients=[p.grad.detach().clone() for p in params] if optimizer=='checked' else None
        torch.nn.utils.clip_grad_norm_(net.parameters(),5); opt.step()
        if optimizer=='checked':
            increments=[p.detach()-x for p,x in zip(params,before)]
            sqnorm=sum(float(g.square().sum()) for g in gradients)
            dot=sum(float((g*d).sum()) for g,d in zip(gradients,increments))
            dnorm=sum(float(d.square().sum()) for d in increments)**.5
            effective_lr=opt.param_groups[0]['lr']
            fallback=dot> -1e-4*effective_lr*sqnorm or dnorm>effective_lr/1e-8*sqnorm**.5
            if fallback:
                increments=[-effective_lr*g for g in gradients]
                dot=-effective_lr*sqnorm
            accepted=False
            with torch.no_grad():
                for trial in range(16):
                    scale=2.**(-trial)
                    for p,x,d in zip(params,before,increments):p.copy_(x+scale*d)
                    proposed=obj.loss(R.initial_transport(obj,net()))*(1+aux)
                    if torch.isfinite(proposed) and float(proposed)<=float(loss.detach())+1e-4*scale*dot+1e-14:
                        accepted=True;break
                if not accepted:
                    for p,x in zip(params,before):p.copy_(x)
            checks.append({'epoch':ep,'candidate_evaluations':trial+1,'accepted':accepted,'fallback':fallback,
                           'loss_before':float(loss.detach()),'loss_after':float(proposed) if accepted else float(loss.detach()),
                           'step_scale':scale if accepted else 0.})
    logits=net(); encoded=R.initial_transport(obj,logits)
    final,history=infer(obj,logits.detach(),net.log_eta,damping,smooth,False)
    terminal_encoder=encoded.detach();terminal_final=final.detach()
    terminal_loss=float(obj.loss(final).detach())
    if checkpoint:
        if terminal_loss<=best_loss:
            best_loss=terminal_loss;best_epoch=epochs
        else:
            net.load_state_dict(best_state)
            logits=net();encoded=R.initial_transport(obj,logits)
            final,history=infer(obj,logits.detach(),net.log_eta,damping,smooth,False)
    stages={'random_encoder':random_encoder,'random_final':random_final,
            'trained_encoder':encoded.detach(),'final':final.detach()}
    if checkpoint:stages.update(terminal_encoder=terminal_encoder,terminal_final=terminal_final)
    metadata={'input_seed':feature_seed,'parameter_seed':seed,'warm_start':starting_provenance,'stage_semantics':{'random_encoder':'phase-one initialized encoder' if starting_state is not None else 'random encoder'},'seed':seed,'epochs':epochs,'steps':steps,'variant':variant,
              'model_revision':MODEL_REVISION,'delta':obj.delta,'q_floor':obj.q_floor,
              'objective_variant':variant+'-GW',
              'log_floor':0.0,'accept_tolerance':0.0,
              'source_sha256':SOURCE_SHA256,
              'threads':threads,'identity_propagation':identity,'lr':lr,'schedule':schedule,'optimizer':optimizer,
              'checkpoint_selection':checkpoint,'selected_epoch':best_epoch,'terminal_loss':terminal_loss,
              'aux_coefficient':aux,'damping':damping,'smooth_inference':smooth,
              'selection':'minimum checkpoint full equal node-degree GW objective' if checkpoint else 'minimum terminal full equal node-degree GW objective',
              'input_variant':'layer-degree-plus-meanP8-random-matched' if RESTORE_DEGREE else 'meanP8-random-degree-disabled','degree_input_enabled':RESTORE_DEGREE,'random_channels':16,'degree_weight_initialization':'zero; all other old parameters retained','random_diffusion_steps':8,
              'unroll_geometry':'row-bounded','template_cost_rule':variant,'per_step_log_odds_bound':float(np.log(2)) if obj.bounded_step else None,
              'parameter_count':sum(p.numel() for p in net.parameters()),
              'aux_coefficient_registered':obj.coefficient,'numerical_audit':copy.deepcopy(obj.audit),
              'seconds':time.perf_counter()-started,'zero_gradient_epochs':zero_epochs,'training_checks':checks,
              'fusion_weights':torch.softmax(net.layer_logits.detach(),0).tolist() if hasattr(net,'layer_logits') else None,
              'trace':trace,'loss_path':loss_path,'encoder_loss_path':encoder_path,
              'final_inference_history':history,
              'etas':(torch.sigmoid(net.log_eta) if smooth else torch.nn.functional.softplus(net.log_eta)).detach().tolist(),
              'parameter_changes':{n:float((p.detach()-initial_state[n]).norm()) for n,p in net.named_parameters()},
              'stages':{s:{'loss':float(obj.loss(T)),'stats':stats(obj,T)} for s,T in stages.items()}}
    return metadata,stages,copy.deepcopy(net.state_dict())

def evaluate(result, stages,truth,mask):
    # Called only after the last parameter update and all output stages exist.
    valid=mask & (truth>=0)
    for s,T in stages.items():
        pred=T.argmax(1).numpy()
        result['stages'][s].update(ari=float(adjusted_rand_score(truth[valid],pred[valid])),
                                  nmi=float(normalized_mutual_info_score(truth[valid],pred[valid])),
                                  labels=pred.tolist())

def main():
    p=argparse.ArgumentParser();p.add_argument('cases',nargs='+')
    p.add_argument('--variants',nargs='+',default=['baseline'])
    p.add_argument('--epochs',type=int,default=600);p.add_argument('--steps',type=int,default=12)
    p.add_argument('--seeds',type=int,default=3);p.add_argument('--seed-start',type=int,default=0)
    p.add_argument('--threads',type=int,default=1);p.add_argument('--identity',action='store_true')
    p.add_argument('--lr',type=float,default=.012);p.add_argument('--schedule',choices=['constant','cosine'],default='constant')
    p.add_argument('--aux-weight',type=float,default=.25);p.add_argument('--optimizer',choices=['adam','checked'],default='adam');p.add_argument('--checkpoint',action='store_true');a=p.parse_args()
    for case in a.cases:
        layers,truth,mask,K=load_case(case)
        for variant in a.variants:
            tag=f'{a.epochs}ep_S{a.steps}_T{a.threads}'+('_identity' if a.identity else '')+('' if a.lr==.012 else f'_lr{a.lr}')+('' if a.schedule=='constant' else '_cosine')+('' if a.aux_weight==.25 else f'_aux{a.aux_weight}')+('' if a.optimizer=='adam' else '_checked')+('_checkpoint' if a.checkpoint else '')
            folder=ROOT/'实验输出'/tag/case/variant;folder.mkdir(parents=True,exist_ok=True)
            for seed in range(a.seed_start,a.seed_start+a.seeds):
                dest=folder/f'seed{seed}.json'
                if dest.exists():continue
                r,Ts,state=optimize(layers,K,seed,a.epochs,a.steps,variant,a.threads,a.identity,a.lr,a.schedule,a.aux_weight,a.optimizer,a.checkpoint)
                evaluate(r,Ts,truth,mask);r['case']=case
                np.savez_compressed(folder/f'seed{seed}_transports.npz',**{s:T.numpy() for s,T in Ts.items()})
                torch.save(state,folder/f'seed{seed}_network.pt')
                temporary=dest.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8')
                temporary.replace(dest)
                print(json.dumps({'case':case,'variant':variant,'seed':seed,
                                  **{k:r['stages']['final'][k] for k in ['loss','ari']},
                                  'seconds':r['seconds']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
