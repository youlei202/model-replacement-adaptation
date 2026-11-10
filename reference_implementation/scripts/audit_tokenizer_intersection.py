from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoConfig, AutoProcessor, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernel_isqed.contracts import (
    MAX_CONTEXT_TOKENS,
    MISTRAL_TOKENIZER_REGEX_FIX,
    RESPONSE_MODELS,
    TRANSFORMERS_COMMIT,
)
from kernel_isqed.data import build_split_manifest
from kernel_isqed.frozen import source_tree_hash
from kernel_isqed.io import CANONICAL_DIR, read_json, write_table
from kernel_isqed.llm import installed_transformers_commit
from kernel_isqed.run_keys import build_expected_run_keys


def render_length(engine, model_id: str, system_prompt: str, user_prompt: str) -> int:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "enable_thinking": False,
    }
    try:
        encoded = engine.apply_chat_template(messages, **kwargs)
    except TypeError as error:
        if "qwen3.5" in model_id.lower() or "gemma-4" in model_id.lower():
            raise RuntimeError(
                "the pinned chat template cannot disable thinking"
            ) from error
        kwargs.pop("enable_thinking")
        encoded = engine.apply_chat_template(messages, **kwargs)
    if isinstance(encoded, dict):
        token_ids = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        token_ids = encoded.input_ids
    else:
        token_ids = encoded
    return len(token_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the registered-model 4096-token intersection"
    )
    parser.add_argument("--revision-lock", type=Path, required=True)
    args = parser.parse_args()
    if installed_transformers_commit() != TRANSFORMERS_COMMIT:
        raise RuntimeError("prepare requires the protocol-pinned Transformers commit")
    revisions = read_json(args.revision_lock)["models"]
    prompts = pd.read_parquet(CANONICAL_DIR / "prompt_manifest.parquet")
    audit_hash = source_tree_hash()
    rows = []
    for model in RESPONSE_MODELS:
        revision = revisions[model.model_id]
        config = AutoConfig.from_pretrained(
            model.model_id, revision=revision, trust_remote_code=True
        )
        if getattr(config, "model_type", "") in {
            "gemma4",
            "gemma4_unified",
            "mistral3",
            "qwen3_5",
        }:
            processor_kwargs = {
                "revision": revision,
                "trust_remote_code": True,
            }
            if getattr(config, "model_type", "") == "mistral3":
                processor_kwargs["fix_mistral_regex"] = MISTRAL_TOKENIZER_REGEX_FIX
            processor = AutoProcessor.from_pretrained(model.model_id, **processor_kwargs)
            engine = processor.tokenizer
        else:
            tokenizer_kwargs = {
                "revision": revision,
                "trust_remote_code": True,
            }
            if model.model_id.startswith("mistralai/"):
                tokenizer_kwargs["fix_mistral_regex"] = MISTRAL_TOKENIZER_REGEX_FIX
            engine = AutoTokenizer.from_pretrained(model.model_id, **tokenizer_kwargs)
        for _, row in prompts.iterrows():
            length = render_length(
                engine, model.model_id, row["system_prompt"], row["user_prompt"]
            )
            rows.append(
                {
                    "model_id": model.model_id,
                    "task_id": row["task_id"],
                    "item_id": row["item_id"],
                    "rendered_tokens": length,
                    "within_limit": length <= MAX_CONTEXT_TOKENS,
                    "source_tree_hash": audit_hash,
                }
            )
    audit = pd.DataFrame(rows)
    eligibility = audit.groupby(["task_id", "item_id"], as_index=False)[
        "within_limit"
    ].all()
    candidates = pd.read_parquet(CANONICAL_DIR / "candidate_manifest.parquet")
    candidates = candidates.merge(eligibility, on=["task_id", "item_id"], how="left")
    candidates["within_tokenizer_intersection"] = (
        candidates["within_limit"].fillna(False).astype(bool)
    )
    write_table(audit, CANONICAL_DIR / "tokenizer_length_audit.parquet")
    split = build_split_manifest(candidates)
    build_expected_run_keys(split)
    print(split.groupby(["task_id", "split"]).size().rename("items"))


if __name__ == "__main__":
    main()
