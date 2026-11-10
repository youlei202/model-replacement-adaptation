# Model Replacement under State and Inference Resource Constraints

Code and experiment results for replacing a retired model using responses from other models and a retained adapter. The experiments distinguish **target-specific retained state** from **inference latency and GPU memory**. An improvement under one constraint does not establish an improvement under the other.

## Main findings

### Retained-state limits

The full-vocabulary study uses Qwen3-8B and Qwen3-14B as targets and all three two-peer combinations from Qwen3-0.6B, Qwen3-1.7B and Qwen3-4B. Every fixed pair receives the same adapter and encoding menu as the joint selector and can retune at every budget.

Methods and comparisons were frozen before collecting **4,096 new document clusters**, equally divided between WikiText103 and the specified arXiv population. Across all eight retained-state caps, **1.7B + 4B remains the selected pair**. The strongest fixed-peer policies and the joint selector use identical deployed states, so the measured peer-reselection gain is exactly zero.

Additional adaptation nevertheless improves distributional fidelity beyond independently reoptimized cheap calibration in five of the six target/pair settings. For the selected pair:

| Target | Cheap calibration: mean TV | Richer adaptation: mean TV | TV improvement | Simultaneous 95% interval |
|---|---:|---:|---:|---:|
| Qwen3-8B | 0.190639 | 0.189706 | 0.000932 | [0.000448, 0.001417] |
| Qwen3-14B | 0.222405 | 0.221439 | 0.000966 | [0.000368, 0.001563] |

Values average the eight declared budgets, from 128 to 2,097,152 **bytes**. Intervals belong to one prespecified 42-contrast family covering both targets and the declared populations. These are modest distributional-fidelity gains, not peer-reselection gains or a claim of improved argmax agreement. All six settings and their zero or negative comparisons remain in the result tables.

The heterogeneous six-model development extension evaluates all 180 target/task/pair combinations. Seventeen of eighteen target/task grids have zero fixed-choice cost; the remaining development estimate is approximately 0.000380 TV. The earlier independent academic-MCQ cohort of 4,891 clusters also matched the fixed policies for both selected targets.

### Inference-resource limits

A separate experiment evaluates frozen policies on **2,768 new academic-MCQ query clusters** for Qwen3.5-9B. At a 70-ms warm-request latency cap:

| Policy, with adapter retuning | TV at 38-GiB allocated-VRAM cap | TV at 40-GiB allocated-VRAM cap |
|---|---:|---:|
| Fixed peers selected by unrestricted fidelity | 0.23064 | 0.19790 |
| Cost-aware fixed peers selected once across the cap grid | 0.21583 | 0.21583 |
| Resource-aware peer selection | 0.21583 | 0.19790 |
| Resource-aware NNLS only | 0.21525 | 0.20596 |

At 38 GiB, the fidelity-selected pair cannot run both models and uses its feasible single-peer fallback. At 40 GiB, the higher-fidelity pair becomes usable. The two corresponding KRR systems measured 41.64/55.68 ms and 32.05/39.17 GiB, including peer inference and final probability computation.

These are warm-resident measurements, not cold-start or network-inclusive guarantees. The two operating points were chosen for explanation after inspecting the complete, previously frozen grid. Some other caps fail live feasibility checks. Across the full grid, the quality-only gain over the cost-aware fixed policy is 0.00224 TV; joint quality/resource success is not established. The grid is a research design, not an external service requirement.

This is an inference-resource selection effect. It does not establish that a few kilobytes of retained target state force a different peer choice. No production requirement or measured target-population scale has been established at which these small target-specific states themselves become the bottleneck.

### Corrected original-family experiments

The corrected MCQ experiments use uniform sampling without replacement over target-independent representatives of eligible, non-ambiguous duplicate clusters, with fixed OOD exclusions. All six models were recollected in one runtime, and full fitting and validation selection were repeated over five overlapping splits. These splits are not five independent datasets.

The original candidate family exhibits a small peer/budget tradeoff for Qwen3.5/MMLU-Pro, with mixed-sign selection gains. Seven of nine original target/task cells have zero selection gain across the reported grid. The Gemma/HellaSwag compact-state result is a compression finding: three-call KRR uses 368–464 actual bits under an 848-bit cap, with mean TV 0.228289, compared with 0.226967 for a 393,392-bit full-precision empirical reference. Matched-call encoded NNLS obtains 0.231006 with 216–224 bits. Compression alone does not establish a serving benefit.

## Repository guide

- [`docs/RESULTS.md`](docs/RESULTS.md): detailed tables, evidence scope and limitations.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md): environments, commands, data requirements and provenance.
- [`results/`](results/): compact numeric outputs from the completed experiments.
- [`aurum/`](aurum/): calibrated anchors, full-support KL/TV training, linear and low-rank adapters, gates, MLPs, literal codecs and independent confirmation.
- [`experiments/`](experiments/): complete-family MCQ studies, new-data collection, resource measurement and full-vocabulary experiments.
- [`benchmarks/`](benchmarks/): corrected sampling, response collection and original-family selection.
- [`reference_implementation/`](reference_implementation/): underlying response, kernel and serialization utilities.

Public feature bases and normalizers are fitted from the selected peers' allowed responses. Target-dependent coefficients are charged to the deployed state. Encoded target bytes, decoded parameters, shared context and peer-model costs are reported separately. Every comparison scores decoded predictions, and fixed and joint policies select from the same candidate objects.

## Running the code

Follow the tested environment instructions and commands in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). CPU tests cover probability objectives, exact zero-residual anchors, live gradients, literal codec accounting, split boundaries and statistical contrasts. Model-response collection requires compatible GPU infrastructure and the pinned model snapshots.

This source release uses neutral package names and portable paths. Numerical result tables are exported from the completed runs. Renaming source files changes their hashes: the published source is a documented derivation, and does not claim to satisfy the original execution locks byte-for-byte. New executions generate new source hashes and method locks. Large response banks and model weights are not embedded in Git; their provenance and availability are described separately.
