# CT-GWAE

Implementation of **CT-GWAE: A Community-Transport Gromov-Wasserstein Autoencoder for Multiplex Community Detection**.

CT-GWAE learns a node–community transport shared across aligned network layers. Layer-specific templates evaluate this common membership, and differentiable GW refinement trains the encoder and update steps under a common structural objective.

## Installation

Use Python 3.12 and a CPU environment:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

For the additional experiment checks, install `requirements_experiments.txt`.

## Quick start

Run commands from the repository root:

```bash
python verify.py --load
python run.py --suite real_world --list
python run.py --dataset aucs --check
python run.py --dataset aucs --diagnostic --output runs/aucs_diagnostic
python evaluate.py --output runs/aucs_diagnostic
```

The diagnostic uses one complete 600-proposal cold start, with 24 training refinement steps and 100 final inference steps. A formal real-network experiment uses ten repeats with ten initializations per repeat:

```bash
python run.py --dataset aucs --workers 4 --output runs/aucs_full
python evaluate.py --output runs/aucs_full
```

`--workers` controls independent single-threaded starts. Training budgets, seeds, partition selection and evaluation masks are stored in `configs/`. Evaluation labels are read after predictions are locked.

## Experiments

| Experiment | Entry |
|---|---|
| Real-network CT-GWAE runs | `run.py --suite real_world` |
| Fixed 320-graph SBM/LFR-derived study | `run.py --suite synthetic` |
| Large sparse-SBM CT-GWAE runs and arXiv | `run.py --suite scalability` |
| Ablations and parameter sensitivity | `experiments/`; see `docs/reproduction.md` |
| Single-CPU timing to 50,000 nodes | `experiments/timing/` |
| Check included reference predictions | `python evaluate.py --reference` |

The public repository contains our numerical implementation and experiment scripts. Third-party baseline sources are linked in `THIRD_PARTY.md`; their copied implementations are not bundled. The main command runs CT-GWAE. Baseline parameter settings and the comparison results remain available as reference records.

## Data

Fixed sparse inputs are included for 358 of the 360 main-suite instances, including all 320 small synthetic graphs and 30 large synthetic graphs. The processed DK and MNIST-derived input arrays are obtained separately. Their dimensions, input hashes, seed settings and original source information remain in `configs/catalog.json` and `docs/datasets.md`.

If you have the exact processed inputs, import them without changing their contents:

```bash
python scripts/import_inputs.py --dataset dkpol --from-folder /path/to/dkpol
python scripts/import_inputs.py --dataset mnist70000 --from-folder /path/to/mnist70000
```

Input loading verifies the frozen hashes. `python verify.py --load` reports the inputs present in this checkout. See `DATA_LICENSES.md` for attribution and terms for the third-party datasets. Model code is MIT licensed; third-party dataset terms are separate.

## Repository layout

- `src/`: CT-GWAE model, loader, training worker and evaluation utilities.
- `configs/`: frozen settings, dataset identities and seeds.
- `data/`: fixed sparse inputs and evaluation masks.
- `experiments/`: ablation, sensitivity and timing scripts.
- `results/`: numerical reference tables and selected predictions.
- `docs/`: reproduction instructions and numerical conventions.

Plotting programs, plotting-data exports, historical training logs and model checkpoints are excluded. The small arrays of twelve Full step parameters are paired inputs for the random-initialization ablation.

## License and citation

Original CT-GWAE code is released under the [MIT License](LICENSE). Third-party data retain their own licenses; see [DATA_LICENSES.md](DATA_LICENSES.md). The author list and software citation metadata are in [CITATION.cff](CITATION.cff).
