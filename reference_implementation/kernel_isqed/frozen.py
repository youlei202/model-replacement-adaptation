from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from .adapters import AdapterContext, AdapterState
from .audit import AuditCell
from .codec import EncodedState, decode_state
from .contracts import PROTOCOL_VERSION
from .io import (
    CANONICAL_DIR,
    PROJECT_ROOT,
    read_json,
    sha256_bytes,
    sha256_file,
    stable_hash,
)


# Downstream analysis/rendering scripts must not invalidate the frozen-state
# provenance. ``source_tree_hash`` therefore covers only the scripts that can
# change the generated states, responses, or unlocks.
ANALYSIS_ONLY_SCRIPTS = {"build_artifacts.py"}


def source_tree_hash() -> str:
    files = []
    for root in (
        PROJECT_ROOT / "kernel_isqed",
        PROJECT_ROOT / "experiments",
        PROJECT_ROOT / "scripts",
    ):
        for path in sorted(root.glob("*.py")):
            if path.name in ANALYSIS_ONLY_SCRIPTS:
                continue
            files.append((str(path.relative_to(PROJECT_ROOT)), sha256_file(path)))
    runner = PROJECT_ROOT / "runner.py"
    files.append((str(runner.relative_to(PROJECT_ROOT)), sha256_file(runner)))
    return stable_hash(files)


def fit_validation_response_hash(
    path: Path = CANONICAL_DIR / "matched_responses.parquet",
) -> str:
    frame = pd.read_parquet(path)
    required = {"model_id", "task_id", "item_id", "split"}
    if not required.issubset(frame.columns):
        raise RuntimeError("matched response release omits registered keys")
    frame = frame[frame["split"].isin(("fit", "validation"))].copy()
    if set(frame["split"].astype(str)) != {"fit", "validation"}:
        raise RuntimeError("fit/validation response release is incomplete")
    keys = ["model_id", "task_id", "item_id", "split"]
    if frame.duplicated(keys).any():
        raise RuntimeError("fit/validation response release contains duplicate keys")
    columns = sorted(frame.columns)
    frame = frame.sort_values(keys)[columns].reset_index(drop=True)
    payload = frame.to_csv(
        index=False, float_format="%.17g", lineterminator="\n", na_rep="<NA>",
    )
    return sha256_bytes(payload.encode("utf-8"))


def verify_frozen_inputs(
    freeze: dict[str, object],
    responses_path: Path = CANONICAL_DIR / "matched_responses.parquet",
) -> None:
    files = {
        "protocol_lock_sha256": CANONICAL_DIR / "protocol_lock.json",
        "expected_run_keys_sha256": CANONICAL_DIR / "expected_run_keys.parquet",
        "split_manifest_sha256": CANONICAL_DIR / "split_manifest.parquet",
        "split_contract_sha256": CANONICAL_DIR / "split_contract.parquet",
        "prompt_manifest_sha256": CANONICAL_DIR / "prompt_manifest.parquet",
        "tokenizer_length_audit_sha256": CANONICAL_DIR
        / "tokenizer_length_audit.parquet",
        "requirements_sha256": PROJECT_ROOT / "requirements.txt",
        "python_version_file_sha256": PROJECT_ROOT / ".python-version",
        "pyproject_sha256": PROJECT_ROOT / "pyproject.toml",
    }
    for field, path in files.items():
        if not path.is_file() or sha256_file(path) != freeze.get(field):
            raise RuntimeError("frozen input contract changed: %s" % path)
    if fit_validation_response_hash(responses_path) != freeze.get(
        "fit_validation_responses_sha256"
    ):
        raise RuntimeError("fit/validation response release changed after freezing")


def verify_feature_state_artifacts(manifest: pd.DataFrame) -> None:
    fields = {"condition", "feature_state_path", "feature_state_sha256"}
    if not fields.issubset(manifest.columns):
        raise RuntimeError("frozen-state manifest omits public-feature provenance")
    paths = manifest["feature_state_path"].fillna("").astype(str)
    digests = manifest["feature_state_sha256"].fillna("").astype(str)
    expected = manifest["condition"].astype(str) == "input_tfidf"
    if (paths[expected] == "").any() or (digests[expected] == "").any():
        raise RuntimeError("input_tfidf has no frozen public-feature state")
    if (paths[~expected] != "").any() or (digests[~expected] != "").any():
        raise RuntimeError("public-feature state is attached to an unrelated condition")
    for relative, digest in set(zip(paths[expected], digests[expected])):
        path = (PROJECT_ROOT / relative).resolve()
        if not path.is_file() or PROJECT_ROOT.resolve() not in path.parents:
            raise RuntimeError(
                "frozen public-feature state is missing or leaves the repository"
            )
        if sha256_file(path) != digest:
            raise RuntimeError("frozen public-feature state digest mismatch")


def verify_test_unlock(freeze_path: Path, freeze: dict[str, object]) -> None:
    unlock_path = CANONICAL_DIR / "test_unlock.json"
    if not unlock_path.is_file():
        raise RuntimeError("formal evaluation is unavailable before the test unlock")
    unlock = read_json(unlock_path)
    if (
        unlock.get("unlock_count") != 1
        or unlock.get("state_freeze_sha256") != sha256_file(freeze_path)
        or unlock.get("source_tree_hash") != freeze.get("source_tree_hash")
    ):
        raise RuntimeError("test unlock does not match the frozen state release")


def _selector_matches(values: pd.Series, value: Any) -> pd.Series:
    text = values.astype(str)
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = pd.to_numeric(text, errors="coerce")
        return numeric == float(value)
    return text == str(value)


class FrozenStateStore:
    def __init__(
        self, manifest_path: Path = CANONICAL_DIR / "serialized_states.parquet"
    ):
        exploration = os.environ.get("KERNEL_ISQED_EXPLORATION") == "1"
        if not manifest_path.is_file():
            raise RuntimeError("serialized-state manifest is missing: %s" % manifest_path)
        self.manifest = pd.read_parquet(manifest_path)
        verify_feature_state_artifacts(self.manifest)
        if exploration:
            self.freeze = {}
            return
        freeze_path = CANONICAL_DIR / "state_freeze.json"
        if not freeze_path.is_file() or not manifest_path.is_file():
            raise RuntimeError(
                "formal evaluation requires the pre-test frozen-state release"
            )
        freeze = read_json(freeze_path)
        if freeze.get("protocol") != PROTOCOL_VERSION:
            raise RuntimeError("frozen-state protocol version does not match the code")
        if sha256_file(manifest_path) != freeze.get("serialized_states_sha256"):
            raise RuntimeError(
                "serialized-state manifest changed after the test unlock"
            )
        if source_tree_hash() != freeze.get("source_tree_hash"):
            raise RuntimeError(
                "source code changed after state freezing; publish a new protocol version"
            )
        verify_test_unlock(freeze_path, freeze)
        verify_frozen_inputs(freeze)
        self.freeze = freeze
        os.environ["KERNEL_ISQED_POST_UNLOCK"] = "1"

    def load(
        self,
        cell: AuditCell,
        context: AdapterContext,
        family: str,
        adapter: str,
        **selectors: Any,
    ) -> Tuple[Optional[AdapterState], pd.Series]:
        selected = self.manifest[
            (self.manifest["audit_cell"].astype(str) == cell.cell_id)
            & (self.manifest["state_family"].astype(str) == family)
            & (self.manifest["adapter"].astype(str) == adapter)
        ]
        for key, value in selectors.items():
            if key not in selected:
                raise KeyError("frozen-state selector is not in the manifest: %s" % key)
            selected = selected[_selector_matches(selected[key], value)]
        if len(selected) != 1:
            raise RuntimeError(
                "expected one frozen state for %s/%s/%s %s; found %d"
                % (cell.cell_id, family, adapter, selectors, len(selected))
            )
        row = selected.iloc[0]
        if str(row["status"]) == "no_feasible_configuration":
            return None, row
        path = (PROJECT_ROOT / str(row["state_path"])).resolve()
        if not path.is_file() or PROJECT_ROOT.resolve() not in path.parents:
            raise RuntimeError("frozen state path is missing or leaves the repository")
        bitstream = path.read_bytes()
        digest = sha256_bytes(bitstream)
        if digest != str(row["bitstream_digest"]):
            raise RuntimeError("frozen state bitstream digest mismatch")
        complete_bits = int(row["complete_coded_bits"])
        encoded = EncodedState(
            bitstream,
            complete_bits,
            len(bitstream) * 8 - complete_bits,
            digest,
            str(row["codec"]),
            int(row["rank"]),
        )
        return decode_state(encoded, context), row

    def load_feature_state(self, cell: AuditCell, condition: str) -> dict[str, object]:
        selected = self.manifest[
            (self.manifest["audit_cell"].astype(str) == cell.cell_id)
            & (self.manifest["condition"].astype(str) == condition)
        ]
        pairs = selected[
            ["feature_state_path", "feature_state_sha256"]
        ].drop_duplicates()
        if len(pairs) != 1:
            raise RuntimeError(
                "expected one frozen public-feature state for %s/%s"
                % (cell.cell_id, condition)
            )
        path = PROJECT_ROOT / str(pairs.iloc[0]["feature_state_path"])
        return read_json(path)


def require_frozen_evaluation() -> None:
    exploration = os.environ.get("KERNEL_ISQED_EXPLORATION") == "1"
    FrozenStateStore()
    responses = pd.read_parquet(
        CANONICAL_DIR / "matched_responses.parquet", columns=["split"]
    )
    if exploration:
        return
    if set(responses["split"].astype(str)) != {
        "fit",
        "validation",
        "iid_test",
        "ood_test",
    }:
        raise RuntimeError("formal evaluation requires the complete four-split release")


def require_pretest_fitting() -> None:
    if os.environ.get("KERNEL_ISQED_POST_UNLOCK") == "1":
        raise RuntimeError(
            "adapter fitting and selection are disabled after the one-time test unlock"
        )
