# Third-party methods

This repository distributes original CT-GWAE model and experiment code. It does not redistribute the upstream baseline implementations. Their original licenses and terms apply when acquired from the authors.

| Method | Original source |
|---|---|
| RDSoS | H. Qing, Community detection in multi-layer networks by regularized debiased spectral clustering, https://doi.org/10.1016/j.engappai.2025.110627 |
| SCML | https://web.media.mit.edu/~xdong/code/grassmann.zip |
| WPTTD | https://github.com/frreess/WPTTD |
| DGFM3 | https://github.com/KBergermann/GradFlowModMax |
| jNMF-GCL | https://github.com/YangYiFei112358/jNMF-GCL |
| DMGI | https://github.com/pcy1302/DMGI |
| MultiTensor | https://github.com/cdebacco/MultiTensor |
| InfoMGF-LA | https://github.com/zxlearningdeep/InfoMGF |

`configs/baselines_*.json` retains the settings used for the comparison; `results/` retains numerical reference records. Running third-party methods requires their original implementations and the appropriate input adapter. The root MIT license does not apply to any of these external projects or to third-party datasets.
