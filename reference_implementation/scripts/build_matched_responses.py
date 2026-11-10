from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernel_isqed.contracts import PEERS, RESPONSE_MODELS, validate_model_revision
from kernel_isqed.io import CANONICAL_DIR, read_json, write_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge aligned model response caches")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("fit", "validation", "iid_test", "ood_test"),
        default=("fit", "validation", "iid_test", "ood_test"),
    )
    args = parser.parse_args()
    cache_dir = CANONICAL_DIR / "response_cache"
    frames = []
    for model in RESPONSE_MODELS:
        path = cache_dir / (model.model_id.replace("/", "--") + ".parquet")
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        frames.append(frame[frame["split"].isin(args.splits)])
    matched = pd.concat(frames, ignore_index=True)
    protocol = read_json(CANONICAL_DIR / "protocol_lock.json")
    for model in RESPONSE_MODELS:
        revisions = set(matched.loc[matched["model_id"] == model.model_id, "revision"].astype(str))
        if len(revisions) != 1:
            raise RuntimeError("model cache mixes revisions: %s" % model.model_id)
        validate_model_revision(model.model_id, next(iter(revisions)), protocol)
    key_columns = ["task_id", "item_id", "split"]
    if matched.duplicated(["model_id"] + key_columns).any():
        raise RuntimeError("model response cache contains duplicate keys")
    reference = set(map(tuple, matched[matched["model_id"] == PEERS[0].model_id][key_columns].to_numpy()))
    metadata = [
        "task_id",
        "item_id",
        "split",
        "item_cluster",
        "domain",
        "gold",
        "prompt_hash",
    ]
    reference_metadata = (
        matched[matched["model_id"] == PEERS[0].model_id][metadata]
        .sort_values(key_columns)
        .reset_index(drop=True)
    )
    for model in RESPONSE_MODELS[1:]:
        observed = set(map(tuple, matched[matched["model_id"] == model.model_id][key_columns].to_numpy()))
        if observed != reference:
            raise RuntimeError("model response cache keys do not match: %s" % model.model_id)
        observed_metadata = (
            matched[matched["model_id"] == model.model_id][metadata]
            .sort_values(key_columns)
            .reset_index(drop=True)
        )
        if not observed_metadata.equals(reference_metadata):
            raise RuntimeError("model cache metadata does not match: %s" % model.model_id)
    counts = matched.groupby(["model_id", "task_id", "split"]).size().unstack(fill_value=0)
    output = CANONICAL_DIR / "matched_responses.parquet"
    write_table(matched, output)
    print(counts)
    print("saved:", output)


if __name__ == "__main__":
    main()
