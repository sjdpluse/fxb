# Source, attribution and dataset terms

## Original computational model

`src/fxb/upstream/shiu_model.py` is an unmodified copy of `model.py` from
[philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model),
commit `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`.

- Copyright holders named in the original license: Philip Shiu and Nico Spiller.
- Original MIT license is preserved in `src/fxb/upstream/LICENSE`.
- Original README is preserved in `src/fxb/upstream/README.md`.
- Vendored source SHA-256: `cab4ca5fd1a7aa56a16271eb76d556c462e1fe395a4b9f517bb3ab50492e956d`.
- Publication: [A Drosophila computational brain model reveals sensorimotor processing](https://www.nature.com/articles/s41586-024-07763-9).

FXB adds a separate input and policy adapter; it does not claim the authors
developed, endorsed or validated a trading system. The original neuronal equations,
published graph and parameters are preserved. The adapter changes which neurons
receive external Poisson stimulation, uses count-only spike monitoring, and pools
activity for a learned readout. These are engineering choices, not biological
evidence of financial reasoning.

## FlyWire v783 data

The installer downloads the original repository's `Completeness_783.csv` and
`Connectivity_783.parquet` at the exact commit above. File sizes and SHA-256 digests
are declared in `src/fxb/connectome.py` and checked before use. The original data
are not committed to this repository.

The published v783 model used here contains 138,639 neurons, 15,091,983 weighted
directed connections and 54,492,922 synaptic contacts. This is the complete graph
in those model files, not a claim to include every neuron of every anatomical
release. FlyWire anatomical coordinates are downloaded from the official v783
Codex release and independently hash checked. Their display is a point cloud,
not a neuron-arbor or connectome-edge reconstruction.

[FlyWire's data guidelines](https://join.flywire.ai/guidelines) identify the public
dataset under **CC BY-NC 4.0**. The original code's MIT license does not replace
these data terms. Distribution, sale or other commercial use of a model/data
bundle requires resolving the applicable dataset permissions; this repository
does not grant those rights. Do not assume a commercial MT5 product is licensed
merely because its adapter code can execute real-account orders.

The screenshot in `docs/images/` is the actual FXB interface rendering these
coordinates and actual Brian2 spike counts, with no connected brokerage account.
It credits the Shiu model and FlyWire v783 on the display.

## Runtime dependencies

Brian2 is an external simulator with its own CeCILL license. NumPy, SciPy,
pandas, PyArrow, joblib, FastAPI, Starlette, Uvicorn and the MetaTrader5 SDK retain
their own licenses. Installation does not transfer ownership or broaden dataset
permissions. No upstream model, dataset or library is represented as an original
FXB scientific discovery.
