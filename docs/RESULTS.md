# Experimental results

Retained-state limits and inference-resource limits answer different questions. The experiments retain both improvements and ties, and compare each budget-aware choice with fixed peers whose adapters are independently retuned under the same constraint.

| Setting | Finding | Evidence population |
| --- | --- | --- |
| Corrected original MCQ family | Peer rankings can change with the bit cap, but selection gains are small and include negative values. | Five overlapping fit/validation/test splits |
| Expanded full-vocabulary family | One fixed pair matches budget-aware selection at every tested state cap. | 4,096 new document clusters |
| Richer adaptation versus cheap calibration | Five of six target/pair settings have positive distributional-fidelity improvements with simultaneous intervals. | The same frozen 4,096-cluster confirmation |
| Inference-resource constraints | Different peer choices are useful at two measured latency/VRAM operating points; the full-grid joint quality/resource criterion is not established. | 2,768 new academic-MCQ query clusters |

## Corrected sampling and complete original peer family

Sampling uses uniform selection without replacement over target-independent representatives of eligible, non-ambiguous duplicate clusters, with fixed out-of-distribution exclusions. Fit, validation and test target the same declared query distribution. The [realized domain counts](../results/corrected_sampling/domain_counts.csv), [label counts](../results/corrected_sampling/label_counts.csv) and [cross-split overlap](../results/corrected_sampling/cross_seed_overlap.csv) are retained. All six models were recollected in one runtime, and fitting and selection were repeated over five split seeds. Comparisons with older predictions therefore also include recollection differences; they do not isolate sampling alone.

For Qwen3.5-9B on MMLU-Pro, the complete two-call peer family is below. Each pair independently chooses its adapter and encoding on validation from KRR, convex mixing and encoded NNLS. The entries average test TV over the five overlapping splits; those splits are not five independent datasets.

| State cap, bits | Qwen3 + Olmo | Qwen3 + Ministral | Olmo + Ministral |
| ---: | ---: | ---: | ---: |
| 1,024 | 0.452413 | 0.380248 | 0.376534 |
| 2,048 | 0.453221 | 0.375918 | 0.376534 |
| 8,192 | 0.447594 | 0.368545 | 0.376534 |
| 16,384 | 0.447142 | 0.368545 | 0.376534 |

The deployed budget-aware policy uses validation choices, not the test-best pair. Its mean TV gains over a fixed pair selected by full-precision validation are **0.002535, −0.001149, 0.001504 and 0.001504**. Against the once-selected mean-budget fixed policy, they are **0.001390, −0.001266, 0.002291 and 0.002291**. Seven of the nine original target/task cells have zero selection gain throughout this grid. [Complete pair results](../results/corrected_sampling/qwen_mmlu_complete_pairs.csv), [all nine cells](../results/corrected_sampling/all_cells_budget_gains.csv), [per-split paired contrasts](../results/corrected_sampling/per_split_paired_contrasts.csv), [KRR-only and other family comparisons](../results/corrected_sampling/all_q2_peer_pair_scores_per_seed.csv) and [predetermined zero-state references](../results/corrected_sampling/predetermined_zero_state_references.csv) are included.

Gemma/HellaSwag provides a compact-state example. Three-call KRR under an 848-bit cap uses **368–464 actual bits**, with mean TV **0.228289**. The full-precision empirical reference uses **393,392 bits**, with mean TV **0.226967**. Matched-call encoded NNLS uses **216–224 bits**, with mean TV **0.231006**. This is a compression result; full-precision KRR is not an irreducible-error floor. See the [compact-state table](../results/corrected_sampling/gemma_hellaswag_compact_summary.csv).

## Expanded full-vocabulary confirmation

The target models are Qwen3-8B and Qwen3-14B. The peers are Qwen3-0.6B, Qwen3-1.7B and Qwen3-4B, with all three two-peer combinations included. Every fixed pair receives the same expanded menu and may retune architecture, objective, regularization, rank and encoding at every budget. Methods, deployed states and comparisons were frozen before collecting **4,096 new document clusters**, equally divided between WikiText103 and the specified arXiv population.

Across all eight state caps, **Qwen3-1.7B + Qwen3-4B remains selected**. The joint selector and strongest fixed-peer policies deploy identical states. Their measured selection gain is exactly zero, rather than an uncertain nonzero estimate. The earlier [academic-MCQ confirmation on 4,891 new clusters](../results/state_constraints/academic_mcq_confirmation_4891.csv) also matches the fixed policies for both selected targets. In the [six-model MCQ development extension](../results/state_constraints/mcq_fixed_choice_price_development.csv), 17 of 18 task/target grids have zero fixed-choice cost; the remaining exploratory estimate is approximately 0.000380 TV.

Richer adaptation nevertheless improves over independently reoptimized cheap calibration in **five of six target/pair settings**. For the selected Qwen3-1.7B + Qwen3-4B pair:

| Target | Cheap calibration: mean TV | Richer adaptation: mean TV | TV improvement | Simultaneous 95% interval |
| --- | ---: | ---: | ---: | --- |
| Qwen3-8B | 0.190639 | 0.189706 | 0.000932 | [0.000448, 0.001417] |
| Qwen3-14B | 0.222405 | 0.221439 | 0.000966 | [0.000368, 0.001563] |

These values average equally over eight declared budgets from 128 to 2,097,152 bytes. The intervals belong to one prespecified 42-contrast family covering both targets, all three pairs, the mixture and both source populations. They are modest, reproducible **distributional-fidelity gains**; they do not establish improved argmax agreement or a gain from peer reselection. [All six settings](../results/state_constraints/main_results.csv), [all paired contrasts](../results/state_constraints/fixed_choice_price.csv), [per-budget policy choices](../results/state_constraints/confirmation_policy_summary.csv) and [per-document losses](../results/state_constraints/confirmation_per_cluster.parquet) remain available.

The anchor/loss study and training-size diagnostics distinguish the useful correction from its generalization limit. The public output basis can fit much larger corrections on small training subsets than transfer to held-out documents. Additional development data and two extra declared seed/split sensitivity runs support the modest adaptation gains; they are not counted as independent confirmation data. The [ablation](../results/state_constraints/anchor_loss_ablation.csv), [learning curves](../results/state_constraints/rich_learning_curve_summary.csv) and [sensitivity results](../results/state_constraints/rich_seed_sensitivity.csv) retain all completed outcomes.

## Inference-resource operating points

The inference-resource study uses **2,768 new academic-MCQ query clusters** for Qwen3.5-9B and frozen policies with adapter retuning. Two explanatory operating points at a **70-ms warm-request latency cap** are:

| Policy | 38-GiB allocated-VRAM cap | 40-GiB allocated-VRAM cap |
| --- | ---: | ---: |
| Fixed peers selected by unrestricted fidelity | 0.23064 | 0.19790 |
| Cost-aware fixed peers selected once across the cap grid | 0.21583 | 0.21583 |
| Resource-aware peer selection | 0.21583 | 0.19790 |
| Resource-aware NNLS only | 0.21525 | 0.20596 |

At 38 GiB, the fidelity-selected interface uses its feasible single-peer fallback; changing the selected peers improves TV by **0.01482**. At 40 GiB, the higher-fidelity two-peer system becomes usable. Its TV improves by **0.01792** over the cost-aware fixed policy and by **0.00806** over resource-aware NNLS. All displayed systems pass the recorded resource checks, including peer inference and final probability computation. The two selected KRR systems measured **41.64 / 55.68 ms** and **32.05 / 39.17 GiB**. Measurements are warm-resident and exclude cold start and network transport.

These two caps were chosen for explanation after inspecting the complete, previously frozen grid. They do not replace the full-grid result: some other caps fail live feasibility checks. The quality-only gain over the cost-aware fixed policy across that grid is **0.00224 TV**, and joint quality/resource success is not established. The [two operating points](../results/inference_resources/selected_38_40_GiB_points.csv), [complete cap outcomes](../results/inference_resources/complete_cap_outcomes.csv), [full-grid comparisons](../results/inference_resources/all_grid_comparators.csv), [probability-endpoint checks](../results/inference_resources/probability_endpoint_cap_checks.csv) and [joint quality/resource outcomes](../results/inference_resources/joint_quality_resource_outcomes.csv) preserve those limits.

## Resource accounting and scope

Encoded target state, decoded parameters, shared bases, peer-model inference and final probability computation are separate quantities. The [full-vocabulary state ledger](../results/state_constraints/public_and_target_state.csv) and [adapter-only measurements](../results/state_constraints/adapter_resource_measurements.csv) report their explicit scopes. In particular, a compact target-state encoding can depend on a much larger public basis. Its compression ratio is not a direct total-serving-memory or latency saving.

No production requirement or measured target-population scale has established that these small target-specific states themselves are the bottleneck. The cap grids are research designs, not external service requirements. The measured inference-resource selection effect is distinct from evidence that a few kilobytes of retained state force a different peer choice.

Source hashes and any table transformations are recorded in [PROVENANCE.csv](../results/PROVENANCE.csv). Full-precision fits are empirical references; repeated splits, budgets, encodings and training seeds are not additional independent query clusters.
