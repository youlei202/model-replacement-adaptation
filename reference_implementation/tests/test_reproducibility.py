from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from kernel_isqed.run_keys import KEY_FAMILIES
from kernel_isqed.contracts import (
    ADAPTERS,
    MIN_DOMAIN_STRATUM,
    MIN_GOLD_STRATUM,
    PREFLIGHT_DETERMINISM_TOLERANCE,
    PREFLIGHT_LOGIT_TOLERANCE,
    PREFLIGHT_NORMALIZATION_TOLERANCE,
    PREFLIGHT_PROBABILITY_TOLERANCE,
    PROTOCOL_VERSION,
    RESPONSE_MODELS,
    TRANSFORMERS_COMMIT,
)
from kernel_isqed.frozen import (
    source_tree_hash,
    verify_feature_state_artifacts,
    verify_frozen_inputs,
)
from kernel_isqed.io import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = PROJECT_ROOT / "data/canonical"
RESULTS_DIR = PROJECT_ROOT / "results/tables"
reproducibility_check = pytest.mark.skipif(
    os.environ.get("KERNEL_ISQED_REPRODUCIBILITY_CHECK") != "1",
    reason="reproducibility checks use outputs from a completed formal run",
)

ARTIFACTS = (
    "figures/main/fig02_controlled_information_paths.pdf",
    "figures/main/fig03_adapter_sensitivity.pdf",
    "figures/main/fig04_state_disclosure_and_swap.pdf",
    "figures/main/fig05_operational_frontier.pdf",
    "tables/main/tab02_audit_contracts.tex",
    "tables/main/tab03_core_results.tex",
    "figures/appendix/figA01_full_residual_ledger.pdf",
    "figures/appendix/figA02_disaggregated_frontiers.pdf",
    "figures/appendix/figA03_generalization.pdf",
    "figures/appendix/figA04_information_source_controls.pdf",
    "figures/appendix/figA05_scaling_and_conditioning.pdf",
    "figures/appendix/figA06_kernel_gauge.pdf",
    "figures/appendix/figA07_runtime_and_failures.pdf",
    "figures/appendix/figA08_adapter_comparisons.pdf",
    "tables/appendix/tabA01_model_inventory.tex",
    "tables/appendix/tabA02_data_splits.tex",
    "tables/appendix/tabA03_adapter_state_contracts.tex",
    "tables/appendix/tabA04_full_results.tex",
    "tables/appendix/tabA05_reconstruction_frontier.tex",
    "tables/appendix/tabA06_control_results.tex",
    "tables/appendix/tabA07_statistical_contrasts.tex",
    "tables/appendix/tabA08_compute_failures_provenance.tex",
    "tables/appendix/tabA09_adapter_comparisons.tex",
    "tables/appendix/tabA10_ood_domain_summary.tex",
)

EXPERIMENT_RESULTS = (
    "exp1_exact_peer_clone.csv",
    "exp2_convex_selector_worlds.csv",
    "exp3_random_truth_table_worlds.csv",
    "exp4_planted_bit_llm_geometry.csv",
    "exp5_llm_adapter_sensitivity.csv",
    "exp6_krr_inverse_reconstruction.csv",
    "exp7_quantized_state_disclosure.csv",
    "exp8_target_state_swap.csv",
    "exp9_error_state_query_frontier.csv",
    "exp10_peer_source_controls.csv",
    "exp11_fit_size_generalization.csv",
    "exp12_kernel_scale_gauge.csv",
    "exp13_peer_roster_replication.csv",
    "exp14_target_role_replication.csv",
)


@reproducibility_check
def test_all_required_artifacts_exist() -> None:
    missing = [path for path in ARTIFACTS if not (PROJECT_ROOT / path).is_file()]
    assert not missing, "missing final artifacts: %s" % missing


@reproducibility_check
def test_final_outputs_have_no_placeholders() -> None:
    forbidden = ("TBD", "FILL_ME", "TODO")
    for relative in ARTIFACTS:
        path = PROJECT_ROOT / relative
        if path.suffix == ".tex":
            text = path.read_text(encoding="utf-8")
            assert not any(token in text for token in forbidden)


@reproducibility_check
def test_artifact_manifest_and_expected_keys_are_complete() -> None:
    manifest = json.loads(
        (PROJECT_ROOT / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["count"] == len(ARTIFACTS)
    assert {row["output_file"] for row in manifest["artifacts"]} == set(ARTIFACTS)
    required = (
        "canonical_view",
        "canonical_view_sha256",
        "source_query",
        "source_query_sha256",
        "render_script",
        "render_script_sha256",
        "expected_keys_view",
        "expected_keys_view_sha256",
        "output_file",
        "output_sha256",
    )
    for row in manifest["artifacts"]:
        assert all(row.get(field) for field in required)
        for path_field, hash_field in (
            ("canonical_view", "canonical_view_sha256"),
            ("source_query", "source_query_sha256"),
            ("render_script", "render_script_sha256"),
            ("expected_keys_view", "expected_keys_view_sha256"),
            ("output_file", "output_sha256"),
        ):
            path = PROJECT_ROOT / row[path_field]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == row[hash_field]
    expected = pd.read_parquet(
        PROJECT_ROOT / "data/canonical/expected_run_keys.parquet"
    )
    assert set(expected["key_family"]) == set(KEY_FAMILIES)
    assert not expected.duplicated(["key_family", "key_id"]).any()


@reproducibility_check
def test_protocol_and_immutable_revisions() -> None:
    protocol = json.loads(
        (CANONICAL_DIR / "protocol_lock.json").read_text(encoding="utf-8")
    )
    assert protocol["protocol_version"] == PROTOCOL_VERSION
    assert protocol["transformers_commit"] == TRANSFORMERS_COMMIT
    revisions = list(protocol["revisions"]["models"].values()) + list(
        protocol["revisions"]["datasets"].values()
    )
    assert revisions and all(
        len(value) == 40 and set(value.lower()) <= set("0123456789abcdef")
        for value in revisions
    )
    secondary = {model["model_id"] for model in protocol["secondary_targets"]}
    assert len(secondary) == 3
    rosters = protocol["role_rotation_rosters"]
    assert len(rosters) == 2
    for roster in rosters:
        peer_ids = {model["model_id"] for model in roster["models"]}
        assert len(peer_ids) == 3 and secondary.isdisjoint(peer_ids)


@reproducibility_check
def test_tokenizer_intersection_precedes_split() -> None:
    audit = pd.read_parquet(CANONICAL_DIR / "tokenizer_length_audit.parquet")
    split = pd.read_parquet(CANONICAL_DIR / "split_manifest.parquet")
    assert set(audit["model_id"].astype(str)) == {
        model.model_id for model in RESPONSE_MODELS
    }
    assert split["within_tokenizer_intersection"].astype(bool).all()
    assert set(split["split_protocol"]) == {
        "tokenizer_intersection_then_whole_domain_ood"
    }


@reproducibility_check
def test_split_domains_and_clusters_are_disjoint() -> None:
    split = pd.read_parquet(CANONICAL_DIR / "split_manifest.parquet")
    active = split[split["split"].isin(("fit", "validation", "iid_test", "ood_test"))]
    assert (
        active["dataset_revision"].astype(str).str.fullmatch(r"[0-9a-fA-F]{40}").all()
    )
    assert not (active.groupby("item_cluster")["split"].nunique() > 1).any()
    for _, values in active.groupby("task_id"):
        ood = set(values[values["split"] == "ood_test"]["domain"].astype(str))
        iid = set(values[values["split"] != "ood_test"]["domain"].astype(str))
        assert not ood.intersection(iid)
        supports = [
            set(values[values["split"] == split]["domain"].astype(str))
            for split in ("fit", "validation", "iid_test")
        ]
        assert all(support == supports[0] for support in supports[1:])


@reproducibility_check
def test_split_strata_and_count_rules() -> None:
    split = pd.read_parquet(CANONICAL_DIR / "split_manifest.parquet")
    active = split[split["split"].isin(("fit", "validation", "iid_test", "ood_test"))]
    assert active.groupby(["task_id", "split", "gold"]).size().min() >= MIN_GOLD_STRATUM
    assert (
        active.groupby(["task_id", "split", "domain"]).size().min()
        >= MIN_DOMAIN_STRATUM
    )
    contract = pd.read_parquet(CANONICAL_DIR / "split_contract.parquet")
    assert {"requested_items", "actual_items", "count_rule"}.issubset(contract.columns)
    iid = contract[contract["split"] == "iid_test"]
    assert (iid["actual_items"].astype(int) == iid["requested_items"].astype(int)).all()


@reproducibility_check
def test_preflight_has_official_equivalence_and_generation_checks() -> None:
    files = sorted((CANONICAL_DIR / "preflight").glob("*.parquet"))
    assert len(files) == 9
    for path in files:
        frame = pd.read_parquet(path)
        assert len(frame) == 256 and set(frame["status"]) == {"completed"}
        assert set(frame["length_bin"].astype(int)) == {0, 1, 2, 3}
        assert frame["tail_anchor"].astype(bool).any()
        assert frame["probability_sum_error"].max() <= PREFLIGHT_NORMALIZATION_TOLERANCE
        assert (
            frame["deterministic_max_difference"].max()
            <= PREFLIGHT_DETERMINISM_TOLERANCE
        )
        assert frame["centered_logit_max_difference"].max() <= PREFLIGHT_LOGIT_TOLERANCE
        assert (
            frame["official_probability_max_difference"].max()
            <= PREFLIGHT_PROBABILITY_TOLERANCE
        )
        assert (frame["peak_allocated_bytes"] < frame["vram_limit_bytes"]).all()
        sanity = frame[frame["generation_valid"].notna()]
        assert len(sanity) == 32


@reproducibility_check
def test_codebooks_and_answer_prefixes_are_frozen() -> None:
    inventories = sorted((CANONICAL_DIR / "preflight").glob("*_inventory.json"))
    assert len(inventories) == 9
    for path in inventories:
        inventory = json.loads(path.read_text(encoding="utf-8"))
        assert inventory["thinking_enabled"] is False
        assert inventory["transformers_commit"] == TRANSFORMERS_COMMIT
        assert inventory["source_tree_hash"] == source_tree_hash()
        assert (
            inventory["scored_position"] == "last_token_of_rendered_generation_prefix"
        )
        assert inventory["final_answer_prefix_ids"]
        assert inventory["code_tokenization"] in {
            "chat_template_continuation_from_frozen_prefix",
            "tokenizer_text_code_after_frozen_prefix",
        }
        if inventory["model_id"] in {"Qwen/Qwen3.5-9B", "google/gemma-4-12B-it"}:
            assert inventory["loader_class"] == "AutoModelForMultimodalLM"
            assert inventory["multimodal_wrapper"] is True
        if inventory["model_id"] == "mistralai/Ministral-3-8B-Instruct-2512-BF16":
            assert inventory["loader_class"] == "AutoModelForImageTextToText"
            assert inventory["multimodal_wrapper"] is True
        modes = {
            book["codebook_mode"]
            for book in inventory["task_codebooks"].values()
        }
        expected_modes = (
            {"trie_sequence_log_probability"}
            if inventory["model_id"] == "mistralai/Mistral-7B-Instruct-v0.3"
            else {"single_token"}
        )
        assert modes == expected_modes


@reproducibility_check
def test_all_state_families_are_frozen() -> None:
    manifest = pd.read_parquet(CANONICAL_DIR / "serialized_states.parquet")
    expected = {
        "frontier_subset",
        "frontier_selected",
        "native",
        "disclosure",
        "swap_control",
        "peer_source_control",
        "fit_size",
        "roster_native",
        "roster_frontier",
        "role_rotation_native",
        "role_rotation_frontier",
    }
    assert expected.issubset(set(manifest["state_family"]))
    assert set(ADAPTERS).issubset(set(manifest["adapter"]))
    verify_feature_state_artifacts(manifest)
    tfidf = manifest[manifest["condition"] == "input_tfidf"]
    assert len(tfidf) > 0
    for row in (
        tfidf[["feature_state_path", "feature_state_sha256"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        path = PROJECT_ROOT / row.feature_state_path
        assert path.is_file() and sha256_file(path) == row.feature_state_sha256
    for row in manifest[manifest["status"] == "frozen"].itertuples(index=False):
        path = PROJECT_ROOT / row.state_path
        assert (
            path.is_file()
            and hashlib.sha256(path.read_bytes()).hexdigest() == row.bitstream_digest
        )


@reproducibility_check
def test_one_time_unlock_matches_state_release() -> None:
    unlock = json.loads(
        (CANONICAL_DIR / "test_unlock.json").read_text(encoding="utf-8")
    )
    freeze_path = CANONICAL_DIR / "state_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    assert unlock["unlock_count"] == 1
    assert (
        unlock["state_freeze_sha256"]
        == hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    )
    assert (
        unlock["source_tree_hash"] == freeze["source_tree_hash"] == source_tree_hash()
    )
    assert freeze["requirements_sha256"] == sha256_file(
        PROJECT_ROOT / "requirements.txt"
    )
    assert freeze["python_version_file_sha256"] == sha256_file(
        PROJECT_ROOT / ".python-version"
    )
    assert freeze["pyproject_sha256"] == sha256_file(PROJECT_ROOT / "pyproject.toml")
    verify_frozen_inputs(freeze)


@reproducibility_check
def test_post_unlock_experiments_do_not_fit_or_select() -> None:
    forbidden = (
        "fit_adapter",
        "select_native_configuration",
        "enumerate_operating_points",
        "select_budget_point",
    )
    for number in range(4, 15):
        path = next((PROJECT_ROOT / "experiments").glob("exp%d_*.py" % number))
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path


@reproducibility_check
def test_selector_ids_are_charged_and_q1_convex_is_canonical() -> None:
    frame = pd.read_csv(RESULTS_DIR / "exp9_error_state_query_frontier.csv")
    selected = frame[frame["adapter"].isin(("val_select_peer", "val_select_pair_mean"))]
    assert set(selected["complete_coded_bits"].astype(int)) == {2}
    assert frame[(frame["adapter"] == "convex") & (frame["q"] == 1)].empty


@reproducibility_check
def test_disclosure_never_mislabels_deployment_as_inverse() -> None:
    frame = pd.read_csv(RESULTS_DIR / "exp7_quantized_state_disclosure.csv")
    assert set(frame["decoder_type"]) == {"declared_deployment_decoder"}


@reproducibility_check
def test_compact_mlp_baseline_is_in_main_results() -> None:
    native = pd.read_csv(RESULTS_DIR / "exp5_llm_adapter_sensitivity.csv")
    frontier = pd.read_csv(RESULTS_DIR / "exp9_error_state_query_frontier.csv")
    assert "mlp" in set(native["adapter"]) and "mlp" in set(frontier["adapter"])


@reproducibility_check
def test_item_level_paired_ledgers_exist() -> None:
    names = (
        "exp5_native_item_losses.parquet",
        "exp8_swap_item_losses.parquet",
        "exp9_frontier_item_losses.parquet",
        "exp10_control_item_losses.parquet",
    )
    for name in names:
        frame = pd.read_parquet(RESULTS_DIR / "item_losses" / name)
        assert not frame.empty and {"item_id", "item_cluster"}.issubset(frame.columns)


@reproducibility_check
def test_ood_intervals_use_domain_blocks() -> None:
    frame = pd.read_csv(RESULTS_DIR / "exp11_fit_size_generalization.csv")
    assert set(frame.loc[frame["split"] == "ood_test", "bootstrap_unit"]) == {
        "domain_block"
    }
    assert set(frame.loc[frame["split"] != "ood_test", "bootstrap_unit"]) == {
        "item_cluster"
    }


@reproducibility_check
def test_planted_primary_family_and_geometry_shift() -> None:
    frame = pd.read_csv(RESULTS_DIR / "exp4_planted_bit_llm_geometry.csv")
    summary = frame[frame["row_kind"] == "summary"]
    primary = summary[summary["inference_tier"] == "simultaneous_primary"]
    descriptive = summary[summary["inference_tier"] == "descriptive"]
    assert (
        len(primary) == 27 and primary["information_lower_bound_per_bit"].notna().all()
    )
    assert set(primary["world_law_scope"]) == {
        "response_geometry_capacity_control"
    }
    assert set(primary["tv_strength"]) == {0.25}
    assert set(primary["budget_cap"]) == {2 ** 18}
    assert set(primary["world_count"]) == {512}
    assert set(primary["registered_points"]) == {27}
    assert set(primary["simultaneous_alpha"]) == {0.05}
    assert set(primary["inference_rule"]) == {
        "world_mean_hoeffding_bonferroni_binary_entropy"
    }
    assert set(primary["configuration_rule"]) == {
        "capacity_first_rank_precision_regularization"
    }
    assert set(primary["recovery_rule"]) == {
        "matched_baseline_basis_projection"
    }
    assert set(primary["planted_law"]) == {
        "adapter_response_basis_rademacher"
    }
    assert primary["geometry_digest"].str.len().eq(64).all()
    observed_passes = int(primary["minimum_success"].sum())
    assert set(primary["primary_family_pass_count"]) == {observed_passes}
    assert set(primary["primary_family_success"]) == {observed_passes >= 14}
    assert primary["bit_error_upper_bound"].between(0, 1).all()
    assert primary["bits_per_planted_bit"].gt(0).all()
    assert descriptive["information_lower_bound_per_bit"].isna().all()
    assert (
        frame.loc[frame["row_kind"] == "world", "natural_perturbation_tv"]
        .between(0, 1)
        .all()
    )


@reproducibility_check
def test_peer_roster_replication_grid() -> None:
    frame = pd.read_csv(RESULTS_DIR / "exp13_peer_roster_replication.csv")
    assert frame["roster_id"].nunique() == 2
    assert set(frame["analysis"]) == {"native", "state_swap", "frontier"}
    counts = frame.groupby(["roster_id", "analysis"]).size().to_dict()
    assert set(counts.values()) == {27, 54, 81}
    assert set(frame["q"].astype(int)) == {3}
    swaps = frame[frame["analysis"] == "state_swap"]
    assert len(swaps) == 108
    assert swaps[["delta_swap", "ci_lower", "ci_upper"]].notna().all().all()


@reproducibility_check
def test_target_role_replication_grid() -> None:
    frame = pd.read_csv(RESULTS_DIR / "exp14_target_role_replication.csv")
    assert frame["target_id"].nunique() == 3
    assert frame["roster_id"].nunique() == 2
    assert set(frame["analysis"]) == {"native", "state_swap", "frontier"}
    counts = frame.groupby(["roster_id", "analysis"]).size().to_dict()
    assert set(counts.values()) == {27, 54, 81}
    assert set(frame["q"].astype(int)) == {3}
    swaps = frame[frame["analysis"] == "state_swap"]
    assert len(swaps) == 108
    assert swaps[["delta_swap", "ci_lower", "ci_upper"]].notna().all().all()


@reproducibility_check
def test_krr_inverse_and_swap_evidence_are_complete() -> None:
    inverse = pd.read_csv(RESULTS_DIR / "exp6_krr_inverse_reconstruction.csv")
    swap = pd.read_csv(RESULTS_DIR / "exp8_target_state_swap.csv")
    assert len(inverse) == 27 and set(inverse["codec"]) == {
        "native_fp16",
        "native_fp32",
        "native_fp64",
    }
    assert set(swap["condition"]) == {"other_target", "neutral", "row_shuffled_refit"}
    donor = swap[swap["condition"] == "other_target"]
    assert (
        donor[
            ["delta_swap", "donor_target_tv_change", "donor_ci_lower", "donor_ci_upper"]
        ]
        .notna()
        .all()
        .all()
    )


@reproducibility_check
def test_frontier_view_includes_zero_bits_iqr_and_pareto() -> None:
    view = pd.read_parquet(PROJECT_ROOT / "views/fig05_operational_frontier.parquet")
    assert (view["complete_coded_bits"] == 0).any()
    assert "pareto_efficient" in view and view["pareto_efficient"].astype(bool).any()


@reproducibility_check
def test_statistical_contrast_table_is_numeric() -> None:
    table = pd.read_csv(PROJECT_ROOT / "views/tabA07_statistical_contrasts.csv")
    required = {"estimate", "ci_lower", "ci_upper", "paired_clusters", "sampling_unit"}
    assert required.issubset(table.columns) and table["estimate"].notna().all()
    assert (table["paired_clusters"] > 0).all()


@reproducibility_check
def test_all_experiment_csvs_are_formal_outputs() -> None:
    files = [RESULTS_DIR / name for name in EXPERIMENT_RESULTS]
    fatal = {
        "accounting_failure",
        "method_failure",
        "numerical_failure",
        "wrapper_failure",
    }
    for path in files:
        frame = pd.read_csv(path)
        assert not frame.empty and set(frame["run_mode"].astype(str)) == {"full"}
        assert not set(
            frame.get("status", pd.Series(dtype=str)).dropna().astype(str)
        ).intersection(fatal)


@reproducibility_check
def test_model_caches_match_registered_keys_and_provenance() -> None:
    expected = pd.read_parquet(CANONICAL_DIR / "expected_run_keys.parquet")
    protocol = json.loads(
        (CANONICAL_DIR / "protocol_lock.json").read_text(encoding="utf-8")
    )
    robustness_models = {
        model["model_id"] for model in protocol["peers"] + protocol["targets"]
    }
    for model_id, revision in protocol["revisions"]["models"].items():
        safe_name = model_id.replace("/", "--")
        main = pd.read_parquet(
            CANONICAL_DIR / "response_cache" / (safe_name + ".parquet")
        )
        assert set(main["revision"].astype(str)) == {revision}
        provenance = {
            "prompt_hash",
            "preflight_inventory_sha256",
            "state_freeze_sha256",
        }
        assert provenance.issubset(main.columns)
        inventory_path = CANONICAL_DIR / "preflight" / (safe_name + "_inventory.json")
        inventory_sha256 = sha256_file(inventory_path)
        assert set(main["preflight_inventory_sha256"].astype(str)) == {inventory_sha256}
        freeze_sha256 = sha256_file(CANONICAL_DIR / "state_freeze.json")
        test_rows = main[main["split"].isin(("iid_test", "ood_test"))]
        assert set(test_rows["state_freeze_sha256"].astype(str)) == {freeze_sha256}
        main_expected = expected[
            (expected["key_family"] == "MODEL_MAIN")
            & (expected["model_id"] == model_id)
        ]
        main_columns = ["task_id", "item_id", "split"]
        assert set(map(tuple, main[main_columns].to_numpy())) == set(
            map(tuple, main_expected[main_columns].to_numpy())
        )
        if model_id not in robustness_models:
            continue
        robust = pd.read_parquet(
            CANONICAL_DIR / "robustness_cache" / (safe_name + ".parquet")
        )
        assert set(robust["revision"].astype(str)) == {revision}
        assert provenance.issubset(robust.columns)
        assert set(robust["preflight_inventory_sha256"].astype(str)) == {
            inventory_sha256
        }
        assert set(robust["state_freeze_sha256"].astype(str)) == {freeze_sha256}
        robust_expected = expected[
            (expected["key_family"] == "MODEL_ROBUSTNESS")
            & (expected["model_id"] == model_id)
        ]
        robust_columns = main_columns + ["view_id"]
        assert set(map(tuple, robust[robust_columns].to_numpy())) == set(
            map(tuple, robust_expected[robust_columns].to_numpy())
        )
