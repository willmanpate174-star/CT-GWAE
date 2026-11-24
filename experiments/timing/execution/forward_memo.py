"""Bounded memoization for deterministic frozen CPU64 forward, one outer backward.
The frozen unroll and inner high-order differentiation execute on each cache miss.
No RNG, training, candidate selection or optimizer behavior is replaced here.
"""
import copy,hashlib
from collections import OrderedDict
import torch

class _SavedGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,gradients,*parameters):
        ctx.gradients=gradients;ctx.save_for_backward(*parameters)
        return value.clone()
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,upstream):
        _=ctx.saved_tensors  # Trigger standard parameter in-place version check.
        if upstream.requires_grad:raise RuntimeError('Higher-order outer differentiation unsupported')
        return (None,None,*tuple(None if g is None else g*upstream for g in ctx.gradients))

class ForwardMemo:
    def __init__(self,forward,capacity=128):
        self.original=forward;self.capacity=capacity;self.cache=OrderedDict();self.bound_net=None;self.bound_obj=None
        self.fixed_signature=None;self.fixed_digest=None
        self.stats=dict(wrapper_calls=0,cache_hits=0,real_forward_calls=0,real_gradient_calls=0,real_primal_calls=0,real_inner_steps=0)
    @staticmethod
    def digest(net):
        h=hashlib.sha256()
        for name,value in net.state_dict().items():
            a=value.detach().contiguous();h.update(name.encode());h.update(str((a.dtype,tuple(a.shape))).encode());h.update(a.numpy().tobytes())
        return h.hexdigest()
    @staticmethod
    def fixed_state(net,obj,hash_content=False):
        h=hashlib.sha256();signature=[]
        for prefix,owner,exclude in [('net',net,{'spreads','saturation'}),('obj',obj,{'audit','hard_diagnostic'})]:
            for name,value in sorted(vars(owner).items()):
                if name in exclude or name.startswith('_'):continue
                values=[value] if torch.is_tensor(value) else value if isinstance(value,list) and all(torch.is_tensor(v) for v in value) else None
                if values is not None:
                    for i,a in enumerate(values):
                        if a.requires_grad:continue
                        signature.append((prefix,name,i,id(a),a._version,str(a.dtype),str(a.layout),tuple(a.shape)))
                        if hash_content:
                            if a.layout==torch.strided:parts=[a]
                            elif a.layout==torch.sparse_coo:
                                a=a.coalesce();parts=[a.indices(),a.values()]
                            elif a.layout==torch.sparse_csr:parts=[a.crow_indices(),a.col_indices(),a.values()]
                            else:raise ValueError('Unsupported fixed tensor layout')
                            for part in parts:h.update(part.detach().contiguous().numpy().tobytes())
                elif isinstance(value,(int,float,str,bool,type(None))):signature.append((prefix,name,value))
        h.update(repr(signature).encode());return tuple(signature),h.hexdigest()
    @staticmethod
    def audit_replay(obj,before,after):
        # These are the only side effects of frozen model_core.unroll.
        keys=set(before)|set(after)
        if not keys <= {'bounded_step_calls','max_log_odds_change'}:raise RuntimeError('Unknown objective audit side effect')
        if 'bounded_step_calls' in after:obj.audit['bounded_step_calls']=obj.audit.get('bounded_step_calls',0)+after['bounded_step_calls']-before.get('bounded_step_calls',0)
        if 'max_log_odds_change' in after:obj.audit['max_log_odds_change']=max(obj.audit.get('max_log_odds_change',0.),after['max_log_odds_change'])
    def __call__(self,net,obj,steps,gradient):
        if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():raise RuntimeError('Standard enabled grad mode required')
        # Strong references prevent identity reuse; immutable buffers/operators/features and objective required.
        if self.bound_net is not net or self.bound_obj is not obj:
            self.cache.clear();self.bound_net=net;self.bound_obj=obj
            self.fixed_signature,self.fixed_digest=self.fixed_state(net,obj,True)
        current,_=self.fixed_state(net,obj,False)
        if current!=self.fixed_signature:raise RuntimeError('Immutable feature/operator/objective contract violated')
        parameters=tuple(net.parameters());signature=tuple((id(p),p.requires_grad) for p in parameters)
        if any(p.device.type!='cpu' or p.dtype!=torch.float64 for p in parameters):raise ValueError('Frozen CPU float64 only')
        key=(self.digest(net),signature,self.fixed_digest,steps,bool(gradient),torch.is_grad_enabled());self.stats['wrapper_calls']+=1
        if key in self.cache:
            self.stats['cache_hits']+=1;entry=self.cache.pop(key);self.cache[key]=entry
            self.audit_replay(obj,entry['audit_before'],entry['audit_after'])
            net.spreads=copy.deepcopy(entry['spreads']);net.saturation=entry['saturation']
        else:
            before=copy.deepcopy(obj.audit)
            T,history,loss,checks=self.original(net,obj,steps,gradient)
            after=copy.deepcopy(obj.audit)
            grads=None
            if gradient:
                enabled=tuple(p for p in parameters if p.requires_grad)
                actual=torch.autograd.grad(loss,enabled,allow_unused=True,create_graph=False)
                it=iter(actual);grads=tuple(next(it) if p.requires_grad else None for p in parameters)
                grads=tuple(None if g is None else g.detach().clone() for g in grads)
            entry=dict(T=T.detach().clone(),history=copy.deepcopy(history),loss=loss.detach().clone(),checks=copy.deepcopy(checks),grads=grads,audit_before=before,audit_after=after,spreads=copy.deepcopy(net.spreads),saturation=net.saturation)
            self.cache[key]=entry
            if len(self.cache)>self.capacity:self.cache.popitem(last=False)
            self.stats['real_forward_calls']+=1;self.stats['real_gradient_calls' if gradient else 'real_primal_calls']+=1;self.stats['real_inner_steps']+=steps
        loss=_SavedGradient.apply(entry['loss'],entry['grads'],*parameters) if gradient else entry['loss'].clone()
        return entry['T'].clone(),copy.deepcopy(entry['history']),loss,copy.deepcopy(entry['checks'])
