from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernel_isqed.contracts import RESPONSE_MODELS, TASKS
from kernel_isqed.io import read_json, write_json


def checked_sha(repo_id: str, sha: str | None) -> str:
    value = str(sha or "")
    if re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise RuntimeError("Hugging Face did not return a commit SHA for %s" % repo_id)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve and freeze the Hugging Face repository revisions"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "configs/llm_revisions.json",
    )
    parser.add_argument(
        "--base",
        type=Path,
        help="Preserve revisions from an archived lock and resolve only missing entries",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(
            "revision lock already exists; archive the current run before replacing it"
        )

    base = read_json(args.base) if args.base is not None else {}
    registered_models = {model.model_id for model in RESPONSE_MODELS}
    registered_datasets = {task.dataset_id for task in TASKS}
    models = {
        str(repo_id): checked_sha(str(repo_id), str(revision))
        for repo_id, revision in base.get("models", {}).items()
        if str(repo_id) in registered_models
    }
    datasets = {
        str(repo_id): checked_sha(str(repo_id), str(revision))
        for repo_id, revision in base.get("datasets", {}).items()
        if str(repo_id) in registered_datasets
    }
    api = HfApi()
    for model in RESPONSE_MODELS:
        if model.model_id not in models:
            models[model.model_id] = checked_sha(
                model.model_id, api.model_info(model.model_id).sha
            )
    for task in TASKS:
        if task.dataset_id not in datasets:
            datasets[task.dataset_id] = checked_sha(
                task.dataset_id, api.dataset_info(task.dataset_id).sha
            )
    write_json(args.output, {"models": models, "datasets": datasets})
    print("frozen revisions:", args.output)
    for repo_id, revision in models.items():
        print("model  ", revision, repo_id)
    for repo_id, revision in datasets.items():
        print("dataset", revision, repo_id)


if __name__ == "__main__":
    main()
