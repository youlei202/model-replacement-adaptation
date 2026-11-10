from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib.metadata import distribution
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import try_to_load_from_cache
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from isqed.core import ModelUnit

from .contracts import MISTRAL_TOKENIZER_REGEX_FIX, TRANSFORMERS_COMMIT
from .response import scores_to_probabilities


def installed_transformers_commit() -> str:
    direct_url = distribution("transformers").read_text("direct_url.json")
    if not direct_url:
        return ""
    payload = json.loads(direct_url)
    return str(payload.get("vcs_info", {}).get("commit_id", ""))


@dataclass(frozen=True)
class ScoreResult:
    log_scores: np.ndarray
    probabilities: np.ndarray
    code_token_lengths: Tuple[int, ...]


class SemanticOptionScorer(ModelUnit):
    def __init__(self, model_id: str, revision: str, device: str = "cuda"):
        if re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
            raise ValueError("SemanticOptionScorer requires a 40-character commit SHA")
        super().__init__(name=model_id)
        self.model_id = model_id
        self.revision = revision
        self.device = torch.device(device)
        self.transformers_commit = installed_transformers_commit()
        if self.transformers_commit != TRANSFORMERS_COMMIT:
            raise RuntimeError(
                "Transformers must be installed from the protocol-pinned commit"
            )
        config = AutoConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=True
        )
        self.model_type = str(getattr(config, "model_type", ""))
        self.is_multimodal = self.model_type in {
            "gemma4",
            "gemma4_unified",
            "mistral3",
            "qwen3_5",
        }
        self.is_gemma = self.model_type in {"gemma4", "gemma4_unified"}
        self.code_tokenization = "chat_template_continuation_from_frozen_prefix"
        self.processor = None
        if self.is_multimodal:
            processor_kwargs = {
                "revision": revision,
                "trust_remote_code": True,
            }
            if self.model_type == "mistral3":
                processor_kwargs["fix_mistral_regex"] = MISTRAL_TOKENIZER_REGEX_FIX
            self.processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
            self.tokenizer = self.processor.tokenizer
            if self.model_type == "mistral3":
                model_class = AutoModelForImageTextToText
            else:
                try:
                    from transformers import AutoModelForMultimodalLM
                except ImportError as error:
                    raise RuntimeError(
                        "the pinned Transformers build must expose AutoModelForMultimodalLM"
                    ) from error
                model_class = AutoModelForMultimodalLM
        else:
            tokenizer_kwargs = {
                "revision": revision,
                "trust_remote_code": True,
            }
            if model_id.startswith("mistralai/"):
                tokenizer_kwargs["fix_mistral_regex"] = MISTRAL_TOKENIZER_REGEX_FIX
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
            model_class = AutoModelForCausalLM
        self.loader_class = model_class.__name__
        self.model = model_class.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        self._assert_bf16()
        self.backbone = self._backbone()
        if self.is_gemma:
            self.output_head = getattr(self.model, "lm_head", None)
        else:
            self.output_head = self.model.get_output_embeddings()
        if self.output_head is None and self.model_type == "mistral3":
            self.output_head = self.model.language_model.get_output_embeddings()
        if self.output_head is None:
            raise RuntimeError("model wrapper does not expose an output embedding head")

    def _forward(self, input_data):
        system_prompt, user_prompt, codes = input_data
        return self.score(system_prompt, user_prompt, codes).probabilities

    def _assert_bf16(self) -> None:
        dtypes = {
            parameter.dtype
            for parameter in self.model.parameters()
            if parameter.is_floating_point()
        }
        if dtypes != {torch.bfloat16}:
            raise RuntimeError(
                "all floating model parameters must be BF16, found %s"
                % sorted(str(value) for value in dtypes)
            )

    def _backbone(self):
        if self.is_gemma:
            wrapper = getattr(self.model, "model", None)
            backbone = getattr(wrapper, "language_model", None)
            if backbone is not None:
                return backbone
            if wrapper is not None:
                return wrapper
            raise RuntimeError("Gemma wrapper has no model backbone")
        if self.model_type == "mistral3":
            for path in (
                ("model", "language_model", "model"),
                ("model", "language_model"),
                ("language_model", "model"),
                ("model",),
                ("language_model",),
            ):
                backbone = self.model
                for name in path:
                    backbone = getattr(backbone, name, None)
                    if backbone is None:
                        break
                if backbone is not None and backbone is not self.model:
                    return backbone
            raise RuntimeError("Mistral3 wrapper has no causal language-model backbone")
        for name in ("model", "transformer"):
            candidate = getattr(self.model, name, None)
            if candidate is not None and candidate is not self.model:
                return candidate
        raise RuntimeError("cannot locate causal language model backbone")

    def _cached_snapshot_revision(self, filename: str) -> str:
        cached = try_to_load_from_cache(
            self.model_id, filename, revision=self.revision
        )
        if not isinstance(cached, str):
            raise RuntimeError(
                "cannot verify %s at the requested Hugging Face revision" % filename
            )
        parts = Path(cached).parts
        try:
            snapshot = parts[parts.index("snapshots") + 1]
        except (ValueError, IndexError) as error:
            raise RuntimeError(
                "cached %s is not stored in a Hugging Face snapshot" % filename
            ) from error
        if snapshot != self.revision:
            raise RuntimeError(
                "%s cache revision differs from the requested commit" % filename
            )
        return snapshot

    def _template_engine(self):
        return self.tokenizer

    def _template_ids(
        self,
        messages,
        add_generation_prompt: bool,
        tensors: bool,
        continue_final_message: bool = False,
    ):
        kwargs: Dict[str, object] = {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
            "return_dict": True,
            "enable_thinking": False,
        }
        if tensors:
            kwargs["return_tensors"] = "pt"
        if continue_final_message:
            kwargs["continue_final_message"] = True
        engine = self._template_engine()
        try:
            encoded = engine.apply_chat_template(messages, **kwargs)
        except TypeError as error:
            if "qwen3.5" in self.model_id.lower() or "gemma-4" in self.model_id.lower():
                raise RuntimeError(
                    "the pinned chat template cannot disable thinking"
                ) from error
            kwargs.pop("enable_thinking")
            encoded = engine.apply_chat_template(messages, **kwargs)
        if isinstance(encoded, dict):
            input_ids = encoded["input_ids"]
        elif hasattr(encoded, "input_ids"):
            input_ids = encoded.input_ids
        else:
            input_ids = encoded
        if not tensors:
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.detach().cpu().tolist()
            if (
                isinstance(input_ids, (list, tuple))
                and len(input_ids) == 1
                and isinstance(input_ids[0], (list, tuple))
            ):
                input_ids = input_ids[0]
            if not isinstance(input_ids, (list, tuple)) or not all(
                isinstance(token_id, (int, np.integer)) for token_id in input_ids
            ):
                raise RuntimeError("chat template did not return a token-id sequence")
        return input_ids

    def render(self, system_prompt: str, user_prompt: str) -> torch.Tensor:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        token_ids = self._template_ids(
            messages, add_generation_prompt=True, tensors=True
        )
        return token_ids.to(self.device)

    def _hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
        hidden = output.last_hidden_state
        if hidden.ndim != 3:
            raise RuntimeError("backbone did not return [batch, sequence, hidden]")
        return hidden

    def _candidate_ids(
        self,
        system_prompt: str,
        user_prompt: str,
        codes: Sequence[str],
        prompt_ids: torch.Tensor,
    ) -> List[List[int]]:
        prefix = prompt_ids[0].detach().cpu().tolist()
        ids = []
        for code in codes:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": code},
            ]
            continuation = self._template_ids(
                messages,
                add_generation_prompt=False,
                tensors=False,
                continue_final_message=True,
            )
            continuation = list(continuation)
            if continuation[: len(prefix)] == prefix:
                candidate = continuation[len(prefix) :]
            else:
                finalized = self._template_ids(
                    messages,
                    add_generation_prompt=False,
                    tensors=False,
                    continue_final_message=False,
                )
                finalized = list(finalized)
                if finalized[: len(prefix)] != prefix:
                    candidate = self.tokenizer.encode(
                        code, add_special_tokens=False
                    )
                    if self.tokenizer.decode(candidate) != code:
                        raise RuntimeError("answer code is not losslessly tokenized")
                    if self.model_type not in {"gemma4", "gemma4_unified"}:
                        finalized_tail = list(finalized)
                        while (
                            finalized_tail
                            and finalized_tail[-1] in self.tokenizer.all_special_ids
                        ):
                            finalized_tail.pop()
                        if finalized_tail[-len(candidate) :] != candidate:
                            raise RuntimeError(
                                "assistant template does not extend the frozen generation prefix"
                            )
                    self.code_tokenization = "tokenizer_text_code_after_frozen_prefix"
                else:
                    candidate = finalized[len(prefix) :]
                    while candidate and candidate[-1] in self.tokenizer.all_special_ids:
                        candidate.pop()
            ids.append(candidate)
        if any(not value for value in ids):
            raise RuntimeError("answer code tokenized to an empty sequence")
        return ids

    def _selected_logits(
        self, hidden: torch.Tensor, token_ids: Sequence[int]
    ) -> torch.Tensor:
        indices = torch.tensor(token_ids, dtype=torch.long, device=hidden.device)
        weights = self.output_head.weight.index_select(0, indices)
        bias = getattr(self.output_head, "bias", None)
        selected_bias = bias.index_select(0, indices) if bias is not None else None
        with torch.inference_mode():
            logits = F.linear(hidden, weights, selected_bias)
        text_config = (
            self.model.config.get_text_config()
            if hasattr(self.model.config, "get_text_config")
            else self.model.config
        )
        softcap = getattr(text_config, "final_logit_softcapping", None)
        if softcap is None:
            softcap = getattr(text_config, "logits_soft_cap", None)
        if softcap:
            logits = torch.tanh(logits / float(softcap)) * float(softcap)
        logits = logits.float()
        scaling = getattr(text_config, "logits_scaling", None)
        if scaling is None:
            scaling = getattr(text_config, "logit_scale", None)
        if scaling is not None and not self.is_gemma:
            logits = logits / float(scaling)
        return logits

    def _official_last_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = torch.ones_like(input_ids)
        errors = []
        for keep_name in ("logits_to_keep", "num_logits_to_keep"):
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": False,
                "return_dict": True,
                keep_name: 1,
            }
            try:
                with torch.inference_mode():
                    output = self.model(**kwargs)
                logits = output.logits
                if logits.ndim != 3 or logits.shape[1] != 1:
                    errors.append("official forward did not honor %s=1" % keep_name)
                    continue
                return logits[:, -1, :].float()
            except TypeError as error:
                errors.append(str(error))
        raise RuntimeError(
            "model wrapper cannot return only the final official logit position; "
            "upgrade transformers before running the audit: %s" % " | ".join(errors)
        )

    def _gemma_last_logits_and_hidden(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        captured = []

        def capture_input(module, args):
            del module
            if len(args) != 1 or args[0].ndim != 3:
                raise RuntimeError("Gemma lm_head did not receive [batch, sequence, hidden]")
            captured.append(args[0])

        hook = self.output_head.register_forward_pre_hook(capture_input)
        try:
            logits = self._official_last_logits(input_ids)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError("Gemma official forward did not call lm_head exactly once")
        hidden = captured[0]
        if hidden.shape[1] != 1:
            raise RuntimeError("Gemma official forward did not retain only final hidden state")
        return logits, hidden[:, -1, :]

    def compare_official_logits(
        self, system_prompt: str, user_prompt: str, codes: Sequence[str],
    ) -> Dict[str, float]:
        input_ids = self.render(system_prompt, user_prompt)
        candidates = self._candidate_ids(system_prompt, user_prompt, codes, input_ids)
        branches: Dict[Tuple[int, ...], set[int]] = {}
        for candidate in candidates:
            for index, token_id in enumerate(candidate):
                prefix = tuple(int(value) for value in candidate[:index])
                branches.setdefault(prefix, set()).add(int(token_id))
        comparisons = [
            (prefix, sorted(token_ids))
            for prefix, token_ids in branches.items()
            if len(token_ids) > 1
        ]
        if not comparisons:
            raise RuntimeError("answer codebook has no token-level decision branch")
        centered_differences = []
        probability_differences = []
        for prefix, token_ids in comparisons:
            if prefix:
                suffix = torch.tensor([prefix], dtype=torch.long, device=self.device)
                branch_input_ids = torch.cat([input_ids, suffix], dim=1)
            else:
                branch_input_ids = input_ids
            if self.is_gemma:
                official_logits, hidden = self._gemma_last_logits_and_hidden(
                    branch_input_ids
                )
            else:
                official_logits = self._official_last_logits(branch_input_ids)
                hidden = self._hidden(branch_input_ids)[:, -1, :]
            official = official_logits[0, token_ids]
            selective = self._selected_logits(hidden, token_ids)[0]
            centered_official = official - official.mean()
            centered_selective = selective - selective.mean()
            centered_differences.append(
                torch.max(torch.abs(centered_official - centered_selective))
            )
            probability_differences.append(
                torch.max(
                    torch.abs(
                        torch.softmax(official, dim=-1)
                        - torch.softmax(selective, dim=-1)
                    )
                )
            )
        return {
            "centered_logit_max_difference": float(
                torch.max(torch.stack(centered_differences)).item()
            ),
            "probability_max_difference": float(
                torch.max(torch.stack(probability_differences)).item()
            ),
        }

    def _trie_log_scores(
        self, prompt_ids: torch.Tensor, candidates: Sequence[Sequence[int]]
    ) -> List[float]:
        trie: Dict[object, object] = {}
        for candidate_index, candidate in enumerate(candidates):
            node = trie
            for token_id in candidate:
                node = node.setdefault(int(token_id), {})
            node.setdefault("terminal", []).append(candidate_index)
        scores = [float("-inf")] * len(candidates)

        def visit(
            node: Dict[object, object], prefix: Tuple[int, ...], cumulative: float
        ) -> None:
            for index in node.get("terminal", []):
                scores[int(index)] = cumulative
            children = [
                (int(token), child)
                for token, child in node.items()
                if token != "terminal"
            ]
            if not children:
                return
            if prefix:
                suffix = torch.tensor([prefix], dtype=torch.long, device=self.device)
                input_ids = torch.cat([prompt_ids, suffix], dim=1)
            else:
                input_ids = prompt_ids
            log_probabilities = torch.log_softmax(
                self._official_last_logits(input_ids)[0], dim=-1
            )
            for token_id, child in children:
                visit(
                    child,
                    prefix + (token_id,),
                    cumulative + float(log_probabilities[token_id].item()),
                )

        visit(trie, (), 0.0)
        if not np.isfinite(scores).all():
            raise RuntimeError("trie scorer failed to terminate every answer code")
        return scores

    def score(
        self, system_prompt: str, user_prompt: str, codes: Sequence[str]
    ) -> ScoreResult:
        prompt_ids = self.render(system_prompt, user_prompt)
        candidates = self._candidate_ids(system_prompt, user_prompt, codes, prompt_ids)
        scores = []
        if all(len(candidate) == 1 for candidate in candidates):
            if self.is_gemma:
                _, hidden = self._gemma_last_logits_and_hidden(prompt_ids)
            else:
                hidden = self._hidden(prompt_ids)[:, -1, :]
            logits = self._selected_logits(
                hidden, [candidate[0] for candidate in candidates]
            )[0]
            scores = [float(value.item()) for value in logits]
        else:
            scores = self._trie_log_scores(prompt_ids, candidates)
        log_scores = np.asarray(scores, dtype=np.float64)
        probabilities = scores_to_probabilities(log_scores)[0]
        return ScoreResult(
            log_scores, probabilities, tuple(len(value) for value in candidates)
        )

    def inventory(
        self, system_prompt: str, user_prompt: str, codes: Sequence[str]
    ) -> Dict[str, object]:
        prompt_ids = self.render(system_prompt, user_prompt)
        rendered_ids = prompt_ids[0].detach().cpu().tolist()
        if not rendered_ids:
            raise RuntimeError("chat template has no frozen assistant answer prefix")
        candidate_ids = self._candidate_ids(
            system_prompt, user_prompt, codes, prompt_ids
        )
        template = (
            getattr(self._template_engine(), "chat_template", None)
            or self.tokenizer.chat_template
            or ""
        )
        config_revision = self._cached_snapshot_revision("config.json")
        tokenizer_revision = self._cached_snapshot_revision("tokenizer_config.json")
        return {
            "model_id": self.model_id,
            "requested_revision": self.revision,
            "model_revision": str(config_revision),
            "tokenizer_revision": str(tokenizer_revision),
            "weight_dtype": "bfloat16",
            "transformers_commit": self.transformers_commit,
            "loader_class": self.loader_class,
            "multimodal_wrapper": self.is_multimodal,
            "chat_template_hash": hashlib.sha256(template.encode("utf-8")).hexdigest(),
            "thinking_enabled": False,
            "final_answer_prefix_ids": rendered_ids,
            "final_answer_prefix_hash": hashlib.sha256(
                json.dumps(rendered_ids).encode("utf-8")
            ).hexdigest(),
            "scored_position": "last_token_of_rendered_generation_prefix",
            "code_token_ids": candidate_ids,
            "code_token_lengths": [len(value) for value in candidate_ids],
            "code_tokenization": self.code_tokenization,
            "codebook_mode": "single_token"
            if all(len(value) == 1 for value in candidate_ids)
            else "trie_sequence_log_probability",
            "selective_logit_backend": "verified_final_hidden_plus_output_head",
        }

    def generation_sanity(
        self, system_prompt: str, user_prompt: str, codes: Sequence[str]
    ) -> Dict[str, object]:
        input_ids = self.render(system_prompt, user_prompt)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                do_sample=False,
                max_new_tokens=64,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        pattern = r"(?:^|\s)(%s)(?:\s|$|[\.,:;])" % "|".join(
            re.escape(code) for code in sorted(codes, key=len, reverse=True)
        )
        matches = re.findall(pattern, generated)
        return {
            "generated_text": generated,
            "generation_code": matches[-1] if matches else "",
            "generation_valid": bool(matches),
        }
