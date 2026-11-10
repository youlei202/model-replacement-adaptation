from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernel_isqed.contracts import PROTOCOL_VERSION
from kernel_isqed.frozen import (
    source_tree_hash,
    verify_feature_state_artifacts,
    verify_frozen_inputs,
)
from kernel_isqed.io import (
    CANONICAL_DIR,
    read_json,
    sha256_file,
    stable_hash,
    write_json,
)


def main() -> None:
    freeze_path = CANONICAL_DIR / "state_freeze.json"
    unlock_path = CANONICAL_DIR / "test_unlock.json"
    if not freeze_path.is_file():
        raise RuntimeError("test cannot be unlocked before every state is frozen")
    freeze = read_json(freeze_path)
    if freeze.get("protocol") != PROTOCOL_VERSION:
        raise RuntimeError("state freeze belongs to a different protocol")
    checks = {
        "serialized_states_sha256": CANONICAL_DIR / "serialized_states.parquet",
        "expected_run_keys_sha256": CANONICAL_DIR / "expected_run_keys.parquet",
        "frozen_planted_results_sha256": CANONICAL_DIR
        / "frozen_planted_results.parquet",
        "frozen_planted_oracle_sha256": CANONICAL_DIR.parent
        / "oracle_truth/frozen_planted_openings.parquet",
        "frozen_planted_manifest_sha256": CANONICAL_DIR
        / "frozen_planted_manifest.json",
    }
    for field, path in checks.items():
        if not path.is_file() or sha256_file(path) != freeze.get(field):
            raise RuntimeError("pre-unlock artifact hash mismatch: %s" % path)
    manifest = pd.read_parquet(CANONICAL_DIR / "serialized_states.parquet")
    verify_feature_state_artifacts(manifest)
    verify_frozen_inputs(freeze)
    if source_tree_hash() != freeze.get("source_tree_hash"):
        raise RuntimeError("source tree changed after state freezing")
    responses = pd.read_parquet(
        CANONICAL_DIR / "matched_responses.parquet", columns=["split"]
    )
    if set(responses["split"].astype(str)) != {"fit", "validation"}:
        raise RuntimeError(
            "test responses are already present before the registered unlock"
        )
    freeze_digest = sha256_file(freeze_path)
    if unlock_path.is_file():
        unlock = read_json(unlock_path)
        if (
            unlock.get("unlock_count") != 1
            or unlock.get("state_freeze_sha256") != freeze_digest
        ):
            raise RuntimeError("existing test unlock marker is inconsistent")
        print("test was already unlocked once; resuming the same immutable evaluation")
        return
    write_json(
        unlock_path,
        {
            "protocol": PROTOCOL_VERSION,
            "unlock_count": 1,
            "state_freeze_sha256": freeze_digest,
            "source_tree_hash": source_tree_hash(),
            "unlock_id": stable_hash([freeze_digest, "one_time_test_unlock"]),
            "unlocked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
    print("test unlocked exactly once:", unlock_path)


if __name__ == "__main__":
    main()
