# Reproducing the experiments

The source is organized into four directories:

| Directory | Role |
| --- | --- |
| `benchmarks/` | Corrected sampling, common-runtime response collection, scalar controls and encoded KRR selection |
| `experiments/` | Complete candidate-family experiments, independent confirmation and inference-resource measurements |
| `aurum/` | Full-vocabulary anchor/loss experiments, expanded adapters, learning curves and locked confirmation |
| `reference_implementation/` | Original adapter, state-encoding and controlled-experiment implementations |

`RESULTS.md` describes the numerical findings and the scope of each comparison. Retained target-state limits and inference-resource limits are separate experiments, with different populations and resource definitions.

## Environment

The recorded interpreter was Python 3.11. Core numerical dependencies are pinned in `requirements-analysis.txt`; full-vocabulary training and model inference additionally use `requirements-inference.txt`. The GPU runtime used PyTorch 2.7.1 with CUDA 12.8 on NVIDIA B200 devices. The Transformers implementation is pinned to the exact Git commit used for collection.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-analysis.txt
```

For adapter training and model-response collection, install the appropriate PyTorch build for the host, then:

```bash
python -m pip install -r requirements-inference.txt
```

CPU numerical tests do not require model weights. The `aurum` tests still require PyTorch, even when GPUs are disabled:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest aurum/tests benchmarks/tests experiments/tests experiments/test_confirmation.py -q
```

The frozen historical-cohort integrity test skips when its separately archived data lock is absent; all numerical checks run without that cohort. Tests under `reference_implementation/tests/` additionally import dataset-collection dependencies from `requirements-inference.txt`; their completed-run integrity checks require the original output artifacts.

Use an isolated environment rather than combining arbitrary CUDA package versions from several machines. The requirements files list the experiment dependencies; they are not a claim that all host drivers or CUDA installations yield bitwise-identical outputs.

## Included results and external inputs

Compact result tables and provenance are included in this repository. Full response banks, model weights and large public KRR contexts are separate artifacts. [EXTERNAL_DATA_MANIFEST.json](../results/EXTERNAL_DATA_MANIFEST.json) records the preserved response/context files, sizes and SHA256 identifiers; it is an inventory, not a public download service. Cloning the source alone does not restore those inputs. In particular, the full-vocabulary experiments use FP32 centered-logit arrays with 151,936 columns; each 2,048-document model bank occupies about 1.245 GB.

The following paths describe the input layout expected by the collection and fitting programs:

| Input | Expected location |
| --- | --- |
| Corrected split manifests | `benchmarks/splits/split_seed_*.parquet` |
| Six-model MCQ response banks and provenance | `benchmarks/response_cache/recompute_all/` |
| Pinned full-vocabulary model registry and document metadata | `experiments/runs/v2/rich_behavior/` |
| Original full-vocabulary fit and validation banks | `experiments/runs/v2/rich_behavior/logits/{fit,validation}/model_*.npy` |
| Original peer-only bases and retained baseline states | `experiments/runs/v2/rich_behavior/public_contexts/` and `fits/` |
| Additional development cohort and its five response banks | `runs/aurum/development/new_data/tranche_2048_v1/` |
| Independent confirmation cohort and its five response banks | `runs/aurum/confirmation/cohort_4096_v1/` |

Restore inputs from their original artifact inventory, or regenerate them from the pinned sources and model revisions. Check the declared hashes, row identities, tokenizer semantics, response dimensions and cohort membership before fitting. Access to model checkpoints or source datasets may require their own authentication and terms. Raw source documents and model weights are not bundled here.

The published source uses neutral directory names. Its numerical algorithms derive from the executed source, but path and documentation changes produce different source-file hashes. Original execution hashes identify the original source artifact; they must not be presented as hashes of renamed files. New executions create their own run specifications and source hashes. Numerical result values have not been changed to accommodate the source layout.

[SOURCE_PROVENANCE.csv](SOURCE_PROVENANCE.csv) maps the executed and published source hashes by published file path. Export changes include portable paths and imports, neutral descriptions, a synthetic resource-cap test fixture, and an explicit skip for a historical-data integration test when its external inputs are absent. The numerical algorithms were retained.

## Running the full-vocabulary study

Run commands from the repository root. Use new output directories for each execution and retain completed zero or negative candidates.

After restoring the original fit/validation inputs, a single anchor/loss cell can be run with:

```bash
python -m aurum.run anchor-ablation \
  --target 3 --peers 1 2 --device cuda:0 \
  --output runs/aurum/reproduction/anchor_ablation/t3_p1-2
```

The complete initial family uses targets `3,4` and peer pairs `0 1`, `0 2`, `1 2`. Both fixed and joint policies must receive all six fitted cells and the same encoded candidate catalog.

The expanded menu is implemented by:

```bash
python -m aurum.discover_rich \
  --target 3 --peers 1 2 --device cuda:0 \
  --output runs/aurum/reproduction/expanded/t3_p1-2
```

The corresponding learning-curve runner is `aurum.learning_curve_rich`. It uses registered additional development data, a fixed original peer-only representation, and explicit training-row subsets. Additional fit seeds measure sensitivity on overlapping data; they do not constitute independent confirmation. Consult each command's `--help` before choosing a new run directory.

The controlled fit diagnostics in `aurum.diagnose` deliberately allow memorization on a small set of inner-fit examples. The per-query coefficient diagnostic uses target outputs and is never a deployed predictor or a held-out performance claim.

## Selection and independent confirmation

`aurum.catalog` builds the shared per-pair state frontier and the fixed/native-full, fixed/unbounded-menu, fixed/mean-budget, fixed/minimax-regret, every-pair and joint/cheap policies. Fixed peers remain free to choose a different adapter and encoding at each budget.

`aurum.confirmation freeze` locks the relevant candidate states, policy decisions, population, budget weights, comparison family and inference code. Only after that lock is written does `aurum.confirmation_data prepare` select unused document clusters, and `aurum.confirmation_data collect` collect the pinned model responses. Run these commands with `--help` for their required lock, cohort and cache arguments. The collection command exposes `--cache-root`; no remote host or SSH account is required by the scientific interface.

`aurum.confirmation score` replays the frozen encoded states, writes per-document losses and evaluates the prespecified contrasts. `aurum.confirmation analyze` recomputes the summaries from those losses. The completed study used 4,096 fresh clusters, evenly split between the two named source populations, and one 42-contrast simultaneous family. Whole documents are independent observations; budgets and repeated encodings are not additional samples.

The fixed-choice endpoint is the minimum over fixed pairs of their budget-averaged paired loss relative to joint selection. Simultaneous intervals are formed for all declared paired contrasts before taking the minimum endpoints. In the reported full-vocabulary confirmation, joint and the strongest fixed policies deploy identical states and this cost is exactly zero.

## MCQ and inference-resource experiments

The corrected original MCQ study is implemented by `benchmarks/prepare_splits.py`, `benchmarks/collect_responses.py` and `benchmarks/run_selection.py`. It uses five overlapping split seeds, uniform sampling over eligible duplicate-cluster representatives and a common response-collection runtime. The repeated split results are not five independent datasets.

The complete heterogeneous extension is implemented by `aurum.mcq` and `aurum.mcq_analysis`. The resource-cap study uses `experiments/mcq_resource_cap_*` and the probability-endpoint measurement helpers. Keep its frozen policy grid, single-peer fallback semantics and feasibility checks when recomputing summaries. The displayed 38- and 40-GiB operating points are explanatory points from that grid; they do not supersede the full-grid results.

`aurum.resource_benchmark` measures adapter computation from already-resident peer responses. It includes public projection, correction and final probability computation, but excludes peer-model forward calls. These adapter measurements must not be relabeled as the end-to-end warm-resident timings of the separate resource-cap experiment. Cold start, networks, and a production requirement for small target-state budgets are not established by either a compression ratio or an adapter-only timing.

## Validation and provenance

The numerical tests cover literal codec replay and byte accounting, exact nested anchors, live gradients, full-support losses, train/stopping boundaries, complete-family selection, simultaneous paired inference and data-access safeguards. Archived experimental outputs additionally record their input hashes, model revisions, source hashes, selected states and independent cohort identities.

For a new execution, preserve the full candidate ledger and all signed effects. A probability-fidelity improvement, a compact-state encoding, a peer-reselection effect and a measured inference-resource advantage are separate conclusions and must be checked against their corresponding data and policy definitions.
