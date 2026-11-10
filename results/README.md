# Result tables

The [experimental summary](../docs/RESULTS.md) distinguishes retained target-state budgets, richer adaptation and inference-resource constraints.

- `corrected_sampling/`: five-split MCQ comparisons, complete original peer families, compact-state measurements, realized sampling counts and predetermined zero-state references. Splits overlap and are summarized descriptively.
- `state_constraints/`: frozen full-vocabulary confirmation, every fixed pair and policy, per-document losses, adapter/resource accounting and development diagnostics.
- `inference_resources/`: complete measured latency/VRAM cap grid, its selected explanatory points, full probability-endpoint checks and quality/resource outcomes.
- `PROVENANCE.csv`: original input SHA256, published table SHA256 and transformation description. Numeric outcomes are unchanged; recorded workspace paths and internal module labels are normalized for this repository.

Positive, zero and negative outcomes remain in the tables. Smaller retained-state files do not by themselves imply lower total serving memory or latency. `corrected_sampling/zero_state_input_hashes.csv` identifies the 45 score tables used to assemble the zero-state reference export.
