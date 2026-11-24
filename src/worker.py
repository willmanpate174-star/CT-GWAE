"""One cold model start or one baseline repeat in a separate process."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import argparse
import time
import traceback
from reproduction_common import ROOT, read, save, sha, graph


def model(item, folder, seed_value):
    sys.path.insert(0, str(ROOT/'src/model'))
    from bootstrap import activate
    B, H, C, Schedule, wrapper, decoder, torch, np, cfg, support = activate()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    layers, metadata = graph(item)
    for name in ('proposal_logs', 'proposal_ledgers'):
        (folder/name).mkdir()
    original_save = H.save
    def persist(path, value):
        path = Path(path)
        if path.parent.name in ('proposal_logs', 'proposal_ledgers'):
            with (folder/(path.parent.name+'.jsonl')).open('a', encoding='utf-8') as f:
                import json
                f.write(json.dumps(value, allow_nan=False)+'\n')
            if path.parent.name == 'proposal_logs' and (int(path.stem)+1) % 10 == 0:
                save(folder/'progress.json', dict(completed=int(path.stem)+1, total=600))
        else:
            original_save(path, value)
    H.save = persist
    slot = dict(seed=seed_value, mode='restored', variant='original-control', proposals=600)
    net, obj, training = H.train_candidate(slot, layers, metadata, B, torch,
                                          Schedule, C, folder, cfg, {})
    endpoint = folder/'endpoint'
    endpoint.mkdir()
    raw = B.endpoint(net, obj, endpoint)
    T = np.load(endpoint/'soft_transport.npy', allow_pickle=False)
    original = np.load(endpoint/'labels.npy', allow_pickle=False)
    labels, receipt, raw_score, hard = wrapper.decode_and_score(
        T, original, metadata['K'], obj.hard_diagnostic, decoder, np)
    if abs(raw_score['objective']-raw['hard_objective']) > 1e-10:
        raise ValueError('Hard-objective readback differs')
    if not (np.bincount(labels, minlength=metadata['K']) > 0).all():
        raise ValueError('Known-K decoder omitted a community')
    np.save(folder/'labels.npy', labels, allow_pickle=False)
    save(endpoint/'decoder.json', receipt)
    save(endpoint/'hard_score.json', hard)
    profile = obj.profiles(torch.tensor(T, dtype=torch.float64))
    np.savez_compressed(endpoint/'layer_templates.npz', B=profile['template'].detach().numpy(),
                        q=profile['q'].detach().numpy(), b=profile['b'].detach().numpy(),
                        alpha=profile['alpha'].detach().numpy(), active_layers=np.asarray(obj.active_layers))
    return dict(objective=float(hard['objective']), proposals=600,
                selected_epoch=training['selected_epoch'])


def infomgf(item,folder,seed_value,repeat):
    import numpy as np
    import subprocess
    native=ROOT/'baselines/InfoMGF_LA'
    destination=folder/'native'
    # Training exits before independent inference starts, avoiding retained training memory.
    subprocess.run([sys.executable,'-B',str(native/'train.py'),'--dataset',item['id'],
                    '--repeat',str(repeat),'--seed',str(seed_value),'--output',str(destination)],check=True)
    subprocess.run([sys.executable,'-B',str(native/'replay.py'),'--folder',str(destination),
                    '--dataset',item['id']],check=True)
    s=read(destination/'success.json');v=read(destination/'replay_receipt.json')
    assert s['epochs']==len(s['loss'])==len(s['discriminator_loss'])==100
    assert v['status']=='passed' and v['fresh_process'] and v['exact_embedding']
    for name,key in [('predictions10.npy','predictions_sha256'),('embedding.npy','embedding_sha256'),('checkpoint.pt','checkpoint_sha256')]:
        assert sha(destination/name)==s[key]==v[key]
    predictions=np.load(destination/'predictions10.npy',allow_pickle=False)
    assert predictions.shape==(10,item['N']) and predictions.dtype.kind in 'iu'
    np.save(folder/'labels.npy',predictions,allow_pickle=False)
    return dict(epochs=100,prediction_kind='kmeans10',kmeans_runs=10,
                fresh_process_replay_passed=True,config=s['config'],native_receipts='native/')


def baseline(item, method, folder, seed_value, repeat=0):
    import numpy as np
    if method=='InfoMGF-LA':
        return infomgf(item,folder,seed_value,repeat)
    sys.path.insert(0, str(ROOT/'baselines/runtime'))
    import adapters
    from threadpoolctl import threadpool_limits
    layers, metadata = graph(item)
    if item['suite'] == 'synthetic':
        import synthetic_adapter
        with threadpool_limits(limits=1):
            labels, details = synthetic_adapter.fit(item, method, layers, metadata['K'], seed_value, folder)
        if labels.shape not in ((item['N'],), (item['L'], item['N'])):
            raise ValueError('Invalid original synthetic prediction shape')
        np.save(folder/'labels.npy', labels, allow_pickle=False)
        save(folder/'selection.json', details)
        return details
    config_name = 'baselines_scalability.json' if item['suite']=='scalability' else 'baselines_real.json'
    cfg = read(ROOT/'configs'/config_name)['methods'][method]
    engine = adapters.prepare(method)
    np.random.seed(seed_value)
    with threadpool_limits(limits=1):
        labels, details = adapters.fit(method, layers, metadata['K'], seed_value, cfg, folder, engine)
    labels = np.asarray(labels)
    if labels.dtype.kind not in 'iu' or labels.shape not in ((item['N'],), (item['L'], item['N'])):
        raise ValueError('Invalid baseline prediction')
    np.save(folder/'labels.npy', labels, allow_pickle=False)
    save(folder/'selection.json', details)
    return dict(parameters=cfg, selection=details)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    args = parser.parse_args()
    path = Path(args.job)
    job, folder = read(path), path.parent
    started = time.perf_counter()
    try:
        details = model(job['dataset'], folder, job['seed']) if job['method']=='CT-GWAE' else baseline(job['dataset'], job['method'], folder, job['seed'], job['r'])
        save(folder/'success.json', dict(status='passed', seed=job['seed'], r=job['r'],
             j=job['j'], truth_read=False, sha256=sha(folder/'labels.npy'), **details))
    except BaseException as error:
        save(folder/'failure.json', dict(error=str(error), traceback=traceback.format_exc()))
        raise
    finally:
        save(folder/'elapsed.json', dict(seconds=time.perf_counter()-started))


if __name__ == '__main__':
    main()
