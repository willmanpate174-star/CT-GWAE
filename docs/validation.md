# Local validation — 2026-11-23

Executed on Windows/Python 3.12 in the original pinned CPU environment. No remote job or GitHub publication was submitted.

| Check | Result |
| --- | --- |
| Python source parsing | Passed for every redistributed Python source |
| Main-package hashes, sparse inputs and model imports | Passed: all 360 main-suite inputs loaded |
| Reference-score recomputation | Passed: 5,925 stored prediction arrays, maximum ARI/NMI/ACC difference 0 |
| AUCS diagnostic | Passed: one complete 600-proposal cold start, S24 training, S100 inference, output selection lock and evaluation |
| A ablation numerical preflight | Passed: direct GW contractions, coefficients, gradients/HVP and original Full parity |
| B sensitivity numerical preflight | Passed |
| Fully shared-template preflight | Passed, including its specific shared basis and degenerate supports |
| Node-measure numerical preflight | Passed: all 800 task definitions and the three measures |
| Random-initialization input seal | Passed: all 400 paired Full step-parameter inputs verified |
| Random-initialization execution | Passed: isolated AUCS round, ten random starts and original 100-step kernel |
| 50k timing package | Input/source/plan/environment verification passed; no benchmark training was started |
| Frozen numerical source lineage | Retained and checked by native loaders |

The AUCS diagnostic is not an independent ten-repeat paper reproduction and was not merged into the reference results. Supplementary preflights are numerical checks, not a rerun of every formal experiment. Linux/Slurm execution, fresh dependency installation and complete baseline retraining were not performed in this preparation. Historical checkpoint replay is not newly claimed; checkpoints are intentionally excluded.

The release keeps original numerical kernels byte-identical. Changes are limited to documentation/metadata, package layout, supplementary command routing, removal of private checkpoint preparation/replay from the random-initialization front end, and generic cluster submission parameters. The formal random-initialization kernel itself is unchanged.

## Public distribution checks

The original local validation above used the complete private working input set. Public-checkout verification reports the inputs actually included; DK and MNIST-derived evaluation arrays must be imported separately before their reference scores can be recomputed. The current numerical model files are byte-identical to the frozen model sources. No new full training results are substituted for the paper results.
