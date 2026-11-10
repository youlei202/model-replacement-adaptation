#!/usr/bin/env python3
"""Decode original Table 2 states; old biased split, NOT corrected evidence.

No adapter is fitted. Public eigenbases are reconstructed from original fitting
peer responses, and validation TV must reproduce every frozen value to 1e-6.
"""
from datetime import datetime, timezone
import hashlib
import itertools
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from run_selection import ROOT, PEERS, atomic_json, sha
from selection_core import paired_cluster_interval

BUNDLE = ROOT / "reference_implementation"
sys.path.insert(0, str(BUNDLE))
from kernel_isqed.adapters import make_context, mean_tv, predict_coordinates
from kernel_isqed.codec import EncodedState, decode_state
from kernel_isqed.response import coordinates_to_probabilities, total_variation


def main():
    started = time.monotonic()
    scope = "published_split_diagnostic_not_corrected_evidence"
    response_path = BUNDLE / "data/canonical/matched_responses.parquet"
    state_path = BUNDLE / "data/canonical/serialized_states.parquet"
    response = pd.read_parquet(response_path)
    states = pd.read_parquet(state_path)
    target = "Qwen/Qwen3.5-9B"
    cell = "Qwen--Qwen3.5-9B__mmlu_pro"
    budgets = (1024, 2048, 8192, 16384)
    selected = states[(states.audit_cell == cell) & (states.state_family == "frontier_subset") & (states.adapter == "krr") & (states.q == 2) & states.budget_cap.astype(str).isin([str(b) for b in budgets])].copy()
    if len(selected) != 12:
        raise ValueError("Expected twelve frozen q2 pair/budget states")
    data, metadata = {}, {}
    for split in ("fit", "validation", "iid_test"):
        values = response[(response.task_id == "mmlu_pro") & (response.split == split)]
        aligned = [values[values.model_id == model].sort_values("item_id") for model in PEERS + (target,)]
        if not all(x.item_id.tolist() == aligned[-1].item_id.tolist() for x in aligned):
            raise ValueError("Unmatched original response rows")
        coordinates = [x[[f"coord_{j}" for j in range(9)]].to_numpy(dtype=float) for x in aligned]
        data[split] = (np.stack(coordinates[:3], axis=1), coordinates[3])
        metadata[split] = aligned[-1]
    rows, losses, validation_max_error = [], {}, 0.
    with threadpool_limits(limits=4):
        for pair in itertools.combinations(range(3), 2):
            context = make_context(data["fit"][0], pair)
            subset = selected[selected.peer_indices == "|".join(map(str, pair))]
            for row in subset.itertuples():
                payload = (BUNDLE / row.state_path).read_bytes()
                if hashlib.sha256(payload).hexdigest() != row.bitstream_digest:
                    raise ValueError("Original frozen state digest mismatch")
                encoded = EncodedState(payload, int(row.complete_coded_bits), len(payload) * 8 - int(row.complete_coded_bits), row.bitstream_digest, row.codec, int(row.rank))
                decoded = decode_state(encoded, context)
                validation_prediction = predict_coordinates(decoded, context, data["validation"][0])
                observed_validation = mean_tv(data["validation"][1], validation_prediction)
                delta = abs(observed_validation - row.validation_tv)
                validation_max_error = max(validation_max_error, delta)
                if delta > 1e-6:
                    raise ValueError(f"Original public eigenbasis did not reproduce validation TV: pair={pair}, budget={row.budget_cap}, error={delta}")
                test_prediction = predict_coordinates(decoded, context, data["iid_test"][0])
                loss = total_variation(coordinates_to_probabilities(data["iid_test"][1]), coordinates_to_probabilities(test_prediction))
                budget = int(row.budget_cap)
                losses[(pair, budget)] = loss
                rows.append({"evidence_scope": scope, "target_id": target, "task_id": "mmlu_pro", "budget_bits": budget, "pair_indices": "|".join(map(str, pair)), "pair_models": " + ".join(PEERS[i] for i in pair), "original_meaningful_bits": int(row.complete_coded_bits), "actual_padded_bits": len(payload) * 8, "codec": row.codec, "rank": int(row.rank), "lambda_index": int(row.lambda_index), "frozen_validation_tv": float(row.validation_tv), "reproduced_validation_tv": observed_validation, "test_tv": float(loss.mean()), "state_digest": row.bitstream_digest})
    contrasts = []
    for budget in budgets:
        for left, right in itertools.combinations(tuple(itertools.combinations(range(3), 2)), 2):
            mean, low, high = paired_cluster_interval(losses[(left, budget)] - losses[(right, budget)], metadata["iid_test"].item_cluster.to_numpy(), draws=2000, seed=20261109)
            contrasts.append({"evidence_scope": scope, "budget_bits": budget, "left_pair": "|".join(map(str, left)), "right_pair": "|".join(map(str, right)), "left_test_tv": float(losses[(left, budget)].mean()), "right_test_tv": float(losses[(right, budget)].mean()), "left_minus_right_tv": mean, "ci_low": low, "ci_high": high, "uncertainty": "Paired cluster bootstrap conditional on ORIGINAL biased split and frozen models; unadjusted descriptive interval"})
    output = ROOT / "benchmarks/diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).sort_values(["budget_bits", "pair_indices"])
    table.to_csv(output / "table2_original_pairs.csv", index=False)
    pd.DataFrame(contrasts).to_csv(output / "table2_original_pair_contrasts.csv", index=False)
    atomic_json(output / "table2_original_pairs.json", {"evidence_scope": scope, "purpose": "Recover previously unreported Table 2 cross-pair test comparisons from existing frozen binaries; does not repair sampling or constitute corrected evidence.", "adapter_refitted": False, "validation_reproduction_tolerance": 1e-6, "max_validation_reproduction_error": validation_max_error, "source_hashes": {"responses": sha(response_path), "state_manifest": sha(state_path), "script": sha(Path(__file__)), "original_adapters": sha(BUNDLE / "kernel_isqed/adapters.py"), "original_codec": sha(BUNDLE / "kernel_isqed/codec.py")}, "outputs": {"table2_original_pairs.csv": sha(output / "table2_original_pairs.csv"), "table2_original_pair_contrasts.csv": sha(output / "table2_original_pair_contrasts.csv")}, "runtime": {"python": sys.version, "numpy": np.__version__, "blas_threads": 4, "elapsed_seconds": time.monotonic() - started}, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "command": ".venv-benchmarks/bin/python benchmarks/reproduce_table2.py"})
    print(table[["budget_bits", "pair_indices", "test_tv"]].to_string(index=False))
    print(pd.DataFrame(contrasts)[lambda d: (d.left_pair == "0|2") & (d.right_pair == "1|2")].to_string(index=False))


if __name__ == "__main__":
    main()
