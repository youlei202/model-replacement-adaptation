from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernel_isqed.contracts import RESPONSE_MODELS, TASKS, protocol_dict
from kernel_isqed.data import build_candidate_manifest
from kernel_isqed.io import CANONICAL_DIR, stable_hash, write_json
from kernel_isqed.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the Target-Imprint LLM protocol and data splits"
    )
    parser.add_argument("--output-dir", type=Path, default=CANONICAL_DIR)
    parser.add_argument("--revision-lock", type=Path, required=True)
    args = parser.parse_args()
    revisions = read_json(args.revision_lock)
    expected_models = {model.model_id for model in RESPONSE_MODELS}
    expected_datasets = {task.dataset_id for task in TASKS}
    missing_models = expected_models - set(revisions.get("models", {}))
    missing_datasets = expected_datasets - set(revisions.get("datasets", {}))
    all_revisions = list(revisions.get("models", {}).values()) + list(
        revisions.get("datasets", {}).values()
    )
    if (
        missing_models
        or missing_datasets
        or not all_revisions
        or any(
            re.fullmatch(r"[0-9a-fA-F]{40}", str(value)) is None
            for value in all_revisions
        )
    ):
        raise ValueError(
            "revision lock is incomplete: missing models=%s datasets=%s"
            % (sorted(missing_models), sorted(missing_datasets))
        )
    protocol = protocol_dict()
    protocol["revisions"] = revisions
    protocol["contract_hash"] = stable_hash(protocol)
    write_json(args.output_dir / "protocol_lock.json", protocol)
    manifest = build_candidate_manifest(args.output_dir, revisions["datasets"])
    summary = manifest.groupby("task_id").size().rename("candidate_items")
    print(summary.to_string())
    print(
        "split is intentionally deferred until the registered-model tokenizer intersection is frozen"
    )
    print("saved:", args.output_dir / "protocol_lock.json")


if __name__ == "__main__":
    main()
