"""Execution storage/cache only. Frozen mathematical source files stay unchanged."""
from pathlib import Path
import copy,sys
ROOT=Path(__file__).resolve().parent
class Execution:
    def __init__(self,backend):
        self.B=backend
        self.native_construct=backend.construct
        self.native_forward=backend.forward
        self.memo=None
        self.layouts=None
    def install(self):
        sys.path.insert(0,str(ROOT/"execution"))
        from csr_runtime import apply_csr
        from forward_memo import ForwardMemo
        def construct(*args,**kwargs):
            net,obj,state=self.native_construct(*args,**kwargs)
            # All random initialization/features execute in the original one-thread COO path.
            apply_csr(net,obj,cached_backward=True)
            self.layouts={"operators":[str(x.layout) for x in net.operators],
                          "objective":[str(x.layout) for x in obj.raw]}
            self.memo=ForwardMemo(self.native_forward,capacity=16)
            self.B.forward=self.memo
            return net,obj,state
        self.B.construct=construct
    def stats(self):
        return dict(self.memo.stats) if self.memo is not None else {}
    def report(self):
        return {"layout":self.layouts,"memo_used_in_training":True,"memo_capacity":16,
                "stats":self.stats(),"initialization_threads":1,"training_threads":1,
                "frozen_proposal_loop":600,"no_early_stop":True,
                "independent_replay_memo_used":False}
    def restore(self):
        self.B.construct=self.native_construct
        self.B.forward=self.native_forward
        if self.memo is not None:
            self.memo.cache.clear()
            self.memo.bound_net=None
            self.memo.bound_obj=None

