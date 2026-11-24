"""Read back all trial budgets and replay endpoints using the sealed model."""
from run import ROOT,read,write,sha,adapter,graph


def validate(values,tasks,out,strict_source=True):
    B,H,C,Schedule,wrapper,decoder,torch,np,cfg,support=values
    checks=[]
    for task in tasks:
        folder=out/task['id']/'j00'
        s=read(folder/'success.json');r=s['record'];training=read(folder/'training.json')
        assert r['proposals']==training['proposals']==len(training['trace'])==600
        assert [x['proposal_id'] for x in training['trace']]==list(range(600))
        assert training['actual_training_unroll_steps']==task['S_train']
        assert all(x['actual_training_unroll_steps']==task['S_train'] for x in training['trace'])
        for filename in ('proposal_logs.jsonl','proposal_ledgers.jsonl'):
            with (folder/filename).open(encoding='utf-8') as f:
                ids=[__import__('json').loads(line)['proposal'] for line in f]
            assert ids==list(range(600))
        if strict_source:
            assert s['identity']['source_manifest']==sha(ROOT/'config/source_manifest.json')
        layers,m=graph(task['dataset'],np);proxy=adapter(values,task)
        net,obj,_=proxy.construct(layers,m['K'],r['seed'],'restored','original-control')
        checkpoint=torch.load(folder/'checkpoint.pt',weights_only=False)
        net.load_state_dict(checkpoint['network'])
        T,h,loss,_=proxy.forward(net,obj,100,False)
        saved=np.load(folder/'transport.npy',allow_pickle=False)
        error=float(np.max(np.abs(T.numpy()-saved)));assert error<=1e-12
        labels,receipt,raw,hard=wrapper.decode_and_score(T.numpy(),T.argmax(1).numpy(),m['K'],obj.hard_diagnostic,decoder,np)
        assert np.array_equal(labels,np.load(folder/'labels.npy',allow_pickle=False))
        assert abs(hard['objective']-r['hard_objective'])<=1e-10
        assert abs(float(loss)-r['soft_objective'])<=1e-10
        checks.append(dict(variant=task['variant'],S_train=task['S_train'],seed=r['seed'],
                      proposals=600,terminal_transport_error=error,hard_objective=hard['objective'],
                      labels_sha256=r['labels_sha256']))
    probe=read(ROOT/'full_reference/aucs_probe.json')
    full=next(x for x in checks if x['variant']=='Full' and x['S_train']==24)
    assert full['labels_sha256']==probe['labels_sha256']
    assert abs(full['hard_objective']-probe['hard_objective'])<=1e-10
    result=dict(status='passed',source_manifest=sha(ROOT/'config/source_manifest.json'),
          strict_source_identity=strict_source,cases=checks,formal_results=False,
          scope='600 proposals in each of six trials; saved weights reloaded and S100 replayed with final model')
    write(ROOT/'checks/trial_readback.json',result)
    return result
