"""Fixed-matrix CSR storage adapter; call only AFTER frozen construction.

No objective, initialization, unroll, guards, decoder or optimizer changes.
Only constant COO matrix storage changes; reduction order can differ in float64.
"""
def apply_csr(net, obj, cached_backward=False):
    import torch
    if any(a.requires_grad for a in list(net.operators)+list(obj.raw)):
        raise ValueError('Only fixed sparse matrices are supported')
    if any(a.dtype != torch.float64 or a.device.type != 'cpu'
           for a in list(net.operators)+list(obj.raw)):
        raise ValueError('CPU float64 only')
    net.operators = [a.to_sparse_csr() for a in net.operators]
    obj.raw = [a.to_sparse_csr() for a in obj.raw]
    if cached_backward:
        install_cached_mm(list(net.operators)+list(obj.raw))
    return net, obj


def install_cached_mm(matrices):
    """Cache exact transpose CSR once; fixed linear adjoint supports higher orders.

    Dispatcher acts only on registered constant tensor identities. All other calls
    use the original function. Install once per process after construct; do not
    use on trainable sparse matrices. Returned function restores original mm.
    """
    import torch
    native = getattr(torch.sparse.mm, '_csr_original', torch.sparse.mm)
    lookup={id(a):(a,a.transpose(0,1).to_sparse_csr()) for a in matrices}
    class FixedMM(torch.autograd.Function):
        @staticmethod
        def forward(ctx,a,at,x):
            ctx.a,ctx.at=a,at
            return native(a,x)
        @staticmethod
        def backward(ctx,grad):
            return None,None,FixedMM.apply(ctx.at,ctx.a,grad)
    def mm(a,x,*args,**kwargs):
        item=lookup.get(id(a))
        if item is not None and item[0] is a and not args and not kwargs:
            return FixedMM.apply(item[0],item[1],x)
        return native(a,x,*args,**kwargs)
    mm._csr_original=native
    torch.sparse.mm=mm
    return lambda:setattr(torch.sparse,'mm',native)
