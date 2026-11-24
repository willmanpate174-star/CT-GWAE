# Reproduction guide

Run main commands from the repository root. All fixed input identities, seeds and numerical settings are retained. No plotting scripts, historical training ledgers or checkpoints are shipped.

## Main experiments

```bash
python run.py --suite real_world --dataset aucs --check
python run.py --suite synthetic --dataset all --check
python run.py --suite scalability --dataset sbm_balanced_n1000_g00 --check
python run.py --suite real_world --dataset aucs --method CT-GWAE --output runs/aucs
python run.py --suite synthetic --dataset 000_base_g00 --method CT-GWAE --output runs/synthetic_example
python run.py --suite scalability --dataset sbm_balanced_n1000_g00 --method CT-GWAE --workers 3 --output runs/large_example
```

Third-party baseline sources are acquired from the links in `THIRD_PARTY.md`; their frozen settings are in `configs/`. This public runner executes CT-GWAE. DK and MNIST-derived arrays are imported separately using `scripts/import_inputs.py`.

## Ablation and sensitivity

```bash
python scripts/run_experiment.py --group ablation --list
python scripts/run_experiment.py --group ablation --index 0 --output runs/ablation
python scripts/run_experiment.py --group sensitivity --list
python scripts/run_experiment.py --group sensitivity --index 0 --output runs/sensitivity
python scripts/run_experiment.py --group shared-template --list
python scripts/run_experiment.py --group shared-template --index 0
python scripts/run_experiment.py --group node-measure --list
```

An A/B/shared-template task is one ten-start pool, not one short training. A contains the original mechanism controls and training-depth scan. B contains the occupancy-penalty scan and two directed synthetic controls. The earlier shared-coefficient control is retained under its original name; it is not the later fully shared-template ablation. The latter has its own package and task list.

For node-measure variants enter a Linux/Slurm allocation with at least two CPUs, then use `cd experiments/node_measure; python run.py pair --index 0` (pair indices 0–399). Shared-template pools use their own native `code/school.py pool --index 0` (indices 0–39); their output is stored under experiments/shared_template/results, so --output does not apply to that group. Each pair trains both measures and independently replays its outputs. Node-measure reference metrics are under results/supplementary/node_measure. Native outputs are created locally under the corresponding package results/ directory.

For random initialization:

```bash
cd experiments/random_initialization
python code/study.py check
python code/study.py run --workers 1
```

This uses the original 400 paired arrays of 12 learned Full step parameters, 100 GW steps, no GCN training and the original selection. Small step arrays are fixed experiment inputs, not full model checkpoints. They are necessary to reproduce the matched-budget intervention. The old checkpoint-based Full replay is not included because checkpoints were explicitly excluded. This distinction is recorded rather than pretending to regenerate those parameters from scratch.

Full references, original budgets and seed schedules are included; do not replace the Full comparison with a newly selected best run. Native supplementary summarizers read truth only after prediction selection. Their outputs are separate from the released reference tables.

## Fixed-CPU runtime study

This is a separate experiment from large-graph accuracy: 50 fixed graphs, five independent instances per size/condition, N=1k,5k,10k,20k,50k; one predetermined cold start per graph, 600 proposals, one training thread and independent endpoint replay. CPU time is the sum over protected training proposals, not queue time or multi-start end-to-end elapsed time.

```bash
cd experiments/timing
python run.py verify
export SLURM_PARTITION=your_partition
export SLURM_ACCOUNT=your_account
export SLURM_QOS=your_qos
python run.py submit
```

Submission is an explicit user action and requires Linux/Slurm. It reserves 20 CPUs but runs the 50 training tasks sequentially on one locked CPU. The repository does not automatically submit anything. The original historical account restriction was removed only from submission plumbing; model, task/seed plan, CPU-affinity logic and numerical kernels are retained. Measured times depend on hardware and contention.

## Numerical output/label convention

The frozen experimental runner first computes direct argmax labels. If all K communities are occupied, the wrapper returns those labels byte-for-byte; otherwise the original known-K MAP decoder supplies a surjective assignment before hard-objective selection. The original results must be interpreted with that actual output convention. 

## Result correspondence

- results/real_world: real-network means and available fixed baseline predictions.
- results/synthetic: graph-/condition-level scores and repeat-level baseline evidence.
- results/scalability: large-SBM, MNIST and arXiv references.
- results/supplementary: ablation, sensitivity and matched random-initialization tables.
- results/timing: the completed 50-graph uniform-CPU timing batch; earlier heterogeneous timing batches are not pooled into it.

Figure numbers can change with manuscript revisions; these entries use experimental meaning rather than assuming a fixed figure number.
