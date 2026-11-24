"""One geometric sqrt-degree GW adapter; original cold encoder and mirror routine.
Direct argmax only: no embedding capture, KMeans or hard-label refinement.
"""
from pathlib import Path
import hashlib,importlib.util,sys,time
import numpy as np
import torch
ROOT=Path('C:/Users/Dingjunhao/Desktop/李慧嘉课题组/V8神经网络/实验')
sys.path.insert(0,str(ROOT/'复核_20261005_V8九图成对十重复_v2'))
import backend as OLD
E=OLD.E
state,state_digest=OLD.state,OLD.state_digest
Model=Hard=None


def configure(reg):
    global Model,Hard
    for key,name in [('layerwise_model','geometric_train_model'),('hard_score','geometric_hard_model')]:
        path=Path(reg[key+'_path']).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest()!=reg[key+'_sha256']:raise RuntimeError('Registered '+key+' source binding')
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        if key=='layerwise_model':
            if reg['layerwise_model_class']!='LayerGeometricSingleGW':raise ValueError('Geometric model only')
            Model=module.LayerGeometricSingleGW
        else: Hard=module.LayerGeometricFixedPartition


def construct(layers,k,seed,mode,variant):
    if (mode,variant)!=('restored','original-control'):raise ValueError('Fixed original/restored only')
    E.RESTORE_DEGREE=True
    operators=[E.propagation_operator(layer) for layer in layers]
    features=E.wide_features(layers,operators,seed,k)
    E.R.seed_all(seed);net=E.matched_encoder(operators,features,k,12)
    if Model is None or Hard is None:raise RuntimeError('Registered model/scorer not configured')
    obj=Model(layers,k);obj.hard_diagnostic=Hard(layers,k)
    if obj.delta!=.01 or obj.q_floor!=1/(4*k):raise RuntimeError('Original smoothing/mass contract changed')
    if not all(p.dtype==torch.float64 and p.device.type=='cpu' for p in net.parameters()):raise RuntimeError('CPU float64 encoder required')
    return net,obj,state(net)


def coupling_checks(obj,T):
    if not bool(torch.isfinite(T).all()) or not bool((T>0).all()):raise ValueError('Strict positive API T required')
    row_error=float((T.sum(1)-obj.p).detach().abs().max())
    if row_error>1e-12:raise ValueError('Uniform API row marginal violated')
    W=T/obj.p[:,None];q=obj.source_measures@W
    if not bool(torch.isfinite(q).all()) or not bool((q>0).all()):raise ValueError('Invalid layer target masses')
    layer_checks=[]
    for index,mu in enumerate(obj.source_measures):
        U=mu[:,None]*W
        row=float((U.sum(1)-mu).detach().abs().max());col=float((U.sum(0)-q[index]).detach().abs().max())
        if max(row,col)>1e-12:raise ValueError('Actual layer GW coupling marginal mismatch')
        layer_checks.append(dict(layer_id=obj.active_layers[index],U_row_error=row,U_col_error=col,
            min_layer_q=float(q[index].detach().min()),zero_source_rows=int((mu==0).sum())))
    return dict(min_transport=float(T.detach().min()),min_node_q=float(T.detach().sum(0).min()),
        row_error=row_error,min_layer_q=float(q.detach().min()),coupling_marginals_are_layerwise=True,layers=layer_checks)


def forward(net,obj,steps,gradient):
    if steps not in (24,100):raise ValueError('Fixed S24/S100 contract')
    eta=torch.cat([net.log_eta,net.log_eta[-1].repeat(steps-12)])
    T,history=E.infer(obj,net() if gradient else net().detach(),eta if gradient else eta.detach(),create_graph=gradient)
    if len(history)!=steps:raise RuntimeError('Incomplete mirror horizon')
    checks=coupling_checks(obj,T);loss=obj.loss(T)
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite geometric loss')
    return T,history,loss,checks


def endpoint(net,obj,dest):
    started=time.perf_counter();T,history,loss,checks=forward(net,obj,100,False)
    profile=obj.profiles(T)
    if profile['b'].shape!=(len(obj.active_layers),) or profile['alpha'].shape!=(len(obj.active_layers),):raise RuntimeError('Independent layer coefficients')
    if profile['template'].shape!=(len(obj.active_layers),obj.K,obj.K):raise RuntimeError('Layer B shape')
    labels=T.argmax(1).numpy();counts=np.bincount(labels,minlength=obj.K)
    np.save(dest/'soft_transport.npy',T.numpy());np.save(dest/'labels.npy',labels)
    hard=obj.hard_diagnostic.score(labels)
    if not hard['labels_changed'] is False or hard['active_layers']!=obj.active_layers:raise RuntimeError('Exact unchanged hard scorer/layers')
    (dest/'hard_score.json').write_text(__import__('json').dumps(hard,indent=2,allow_nan=False),encoding='utf-8')
    structure,mass,_=obj.parts(T)
    return dict(soft_objective=float(loss),soft_structure=float(structure),soft_node_mass=float(mass),
        history=history,checks=checks,decoder='direct_argmax_only',decoder_calls=0,KMeans_calls=0,
        labels_counts=counts.tolist(),missing_argmax_communities=int((counts==0).sum()),
        hard_scored=True,hard_objective=hard['objective'],hard_layer_q=[r['q'] for r in hard['profiles']],
        hard_positive_support=[r['m'] for r in hard['profiles']],
        hard_score_domain='unsmoothed onehot, per-layer positive-support Gram m, m1 canonical, full N/K node penalty',
        hard_score_profile=hard,
        layerwise_profile={key:profile[key].detach().tolist() for key in ('q','diagonal_edge_mass','x','b','alpha','gain','layer_structure','b_zero_face','b_zero_threshold_excess')},
        source_profile=dict(mu=obj.source_measures.detach().tolist(),nu=obj.nu.detach().tolist(),null=obj.null.detach().tolist(),zero_volume_layers=obj.zero_volume_layers,inactive_variance_layers=obj.inactive_variance_layers,zero_source_nodes_by_layer=obj.zero_source_nodes_by_layer,relation='raw=A/v; R=raw/(mu_i mu_j) on positive source support'),
        active_layers=obj.active_layers,seconds=time.perf_counter()-started)

