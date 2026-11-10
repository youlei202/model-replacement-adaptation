# runner.py
#
# Resumable multi-GPU runner for the frozen Target-Imprint LLM protocol.
#
# The runner never creates substitute responses or enables --fast. It schedules
# the frozen Hugging Face models on currently available
# GPUs, validates completed caches before skipping them, retries failed jobs,
# and continues with other independent jobs in the same stage.

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from kernel_isqed.contracts import (
    MODEL_JOB_MINIMUM_FREE_GIB,
    PREFLIGHT_DETERMINISM_TOLERANCE,
    PREFLIGHT_LOGIT_TOLERANCE,
    PREFLIGHT_NORMALIZATION_TOLERANCE,
    PREFLIGHT_PROBABILITY_TOLERANCE,
    PREFLIGHT_VRAM_LIMIT_GIB,
    PROTOCOL_VERSION,
    TASKS,
    TRANSFORMERS_COMMIT,
)


PROJECT_ROOT = Path(__file__).resolve().parent
CANONICAL_DIR = PROJECT_ROOT / "data/canonical"
RESULTS_DIR = PROJECT_ROOT / "results/tables"
LOG_DIR = PROJECT_ROOT / "logs/runner"
REVISION_LOCK = PROJECT_ROOT / "configs/llm_revisions.json"

REFERENCE_MODELS = (
    "Qwen/Qwen3-8B",
    "allenai/Olmo-3-7B-Instruct",
    "mistralai/Ministral-3-8B-Instruct-2512-BF16",
    "Qwen/Qwen3.5-9B",
    "google/gemma-4-12B-it",
    "ibm-granite/granite-3.3-8b-instruct",
)

REPLICATION_MODELS = (
    "mistralai/Mistral-7B-Instruct-v0.3",
    "tiiuae/Falcon3-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
)

MODELS = REFERENCE_MODELS + REPLICATION_MODELS

EXPERIMENTS = (
    (1, "experiments/exp1_exact_peer_clone.py", "exp1_exact_peer_clone.csv"),
    (
        2,
        "experiments/exp2_convex_selector_worlds.py",
        "exp2_convex_selector_worlds.csv",
    ),
    (
        3,
        "experiments/exp3_random_truth_table_worlds.py",
        "exp3_random_truth_table_worlds.csv",
    ),
    (
        4,
        "experiments/exp4_planted_bit_llm_geometry.py",
        "exp4_planted_bit_llm_geometry.csv",
    ),
    (
        5,
        "experiments/exp5_llm_adapter_sensitivity.py",
        "exp5_llm_adapter_sensitivity.csv",
    ),
    (
        6,
        "experiments/exp6_krr_inverse_reconstruction.py",
        "exp6_krr_inverse_reconstruction.csv",
    ),
    (
        7,
        "experiments/exp7_quantized_state_disclosure.py",
        "exp7_quantized_state_disclosure.csv",
    ),
    (8, "experiments/exp8_target_state_swap.py", "exp8_target_state_swap.csv"),
    (
        9,
        "experiments/exp9_error_state_query_frontier.py",
        "exp9_error_state_query_frontier.csv",
    ),
    (10, "experiments/exp10_peer_source_controls.py", "exp10_peer_source_controls.csv"),
    (
        11,
        "experiments/exp11_fit_size_generalization.py",
        "exp11_fit_size_generalization.csv",
    ),
    (12, "experiments/exp12_kernel_scale_gauge.py", "exp12_kernel_scale_gauge.csv"),
    (
        13,
        "experiments/exp13_peer_roster_replication.py",
        "exp13_peer_roster_replication.csv",
    ),
    (
        14,
        "experiments/exp14_target_role_replication.py",
        "exp14_target_role_replication.csv",
    ),
)


@dataclass(frozen=True)
class GPU:
    index: int
    free_gb: float
    total_gb: float
    utilization: int


@dataclass
class Job:
    name: str
    command: List[str]
    validator: Callable[[], bool]
    minimum_free_gb: float = 2.0
    needs_gpu: bool = True


class GPUManager:
    def __init__(self, poll_seconds: int, maximum_utilization: int = 20):
        self.poll_seconds = poll_seconds
        self.maximum_utilization = maximum_utilization
        self.in_use: Set[int] = set()
        self.condition = threading.Condition()

    @staticmethod
    def query() -> List[GPU]:
        command = [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.check_output(
                command, text=True, stderr=subprocess.STDOUT
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                "nvidia-smi is unavailable; real LLM jobs require visible NVIDIA GPUs"
            ) from error
        gpus = []
        for line in output.splitlines():
            if not line.strip():
                continue
            index, free_mb, total_mb, utilization = [
                value.strip() for value in line.split(",")
            ]
            gpus.append(
                GPU(
                    int(index),
                    float(free_mb) / 1024.0,
                    float(total_mb) / 1024.0,
                    int(utilization),
                )
            )
        if not gpus:
            raise RuntimeError("nvidia-smi returned no visible GPUs")
        return gpus

    def acquire(self, minimum_free_gb: float) -> GPU:
        with self.condition:
            while True:
                visible = self.query()
                largest = max(gpu.total_gb for gpu in visible)
                if largest < minimum_free_gb:
                    raise RuntimeError(
                        "largest visible GPU has %.1f GiB total memory, below the %.1f GiB job requirement"
                        % (largest, minimum_free_gb)
                    )
                candidates = [
                    gpu
                    for gpu in visible
                    if gpu.index not in self.in_use
                    and gpu.free_gb >= minimum_free_gb
                    and gpu.utilization <= self.maximum_utilization
                ]
                if candidates:
                    selected = max(
                        candidates, key=lambda gpu: (gpu.free_gb, -gpu.utilization)
                    )
                    self.in_use.add(selected.index)
                    return selected
                print(
                    "[WAIT] No GPU currently satisfies %.1f GiB free and <= %d%% utilization."
                    % (minimum_free_gb, self.maximum_utilization),
                    flush=True,
                )
                self.condition.wait(timeout=self.poll_seconds)

    def release(self, gpu: GPU) -> None:
        with self.condition:
            self.in_use.discard(gpu.index)
            self.condition.notify_all()


class TargetImprintRunner:
    def __init__(
        self,
        revisions: Dict[str, str],
        max_retries: int,
        workers: int,
        poll_seconds: int,
        force: bool,
        dry_run: bool,
    ):
        self.revisions = revisions
        self.max_retries = max_retries
        self.workers = workers
        self.force = force
        self.dry_run = dry_run
        self.gpus = GPUManager(poll_seconds)
        self.print_lock = threading.Lock()
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def message(self, text: str) -> None:
        with self.print_lock:
            print(text, flush=True)

    def run_process(self, job: Job, attempt: int, gpu: Optional[GPU]) -> int:
        command = [sys.executable] + job.command
        environment = os.environ.copy()
        if gpu is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu.index)
        log_path = LOG_DIR / (job.name + ".log")
        mode = "a" if log_path.is_file() else "w"
        label = "GPU %d" % gpu.index if gpu is not None else "CPU"
        self.message(
            "[RUN] %s (attempt %d/%d, %s)"
            % (job.name, attempt, self.max_retries + 1, label)
        )
        self.message("      " + " ".join(command))
        if self.dry_run:
            return 0

        with log_path.open(mode, encoding="utf-8") as log:
            log.write("\n=== attempt %d | %s ===\n" % (attempt, label))
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                self.message("[%s] %s" % (job.name, line.rstrip()))
            return process.wait()

    def run_job(self, job: Job) -> Tuple[str, bool]:
        if not self.force and job.validator():
            self.message(
                "[SKIP] %s already has a validated complete output." % job.name
            )
            return job.name, True
        if self.dry_run:
            self.run_process(job, 1, None)
            return job.name, True

        for attempt in range(1, self.max_retries + 2):
            gpu = None
            try:
                gpu = self.gpus.acquire(job.minimum_free_gb) if job.needs_gpu else None
                return_code = self.run_process(job, attempt, gpu)
            except Exception as error:
                return_code = -1
                self.message(
                    "[ERROR] %s raised %s: %s" % (job.name, type(error).__name__, error)
                )
            finally:
                if gpu is not None:
                    self.gpus.release(gpu)
            if return_code == 0 and job.validator():
                self.message(
                    "[OK] %s completed and passed output validation." % job.name
                )
                return job.name, True
            self.message(
                "[ERROR] %s failed or produced an invalid output (exit=%d)."
                % (job.name, return_code)
            )
            if attempt <= self.max_retries:
                self.message(
                    "[RETRY] %s will be retried without restarting successful jobs."
                    % job.name
                )
        return job.name, False

    def run_batch(self, jobs: Sequence[Job]) -> List[str]:
        failures = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self.workers, len(jobs))
        ) as executor:
            futures = {executor.submit(self.run_job, job): job.name for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                submitted_name = futures[future]
                try:
                    name, success = future.result()
                except Exception as error:
                    name, success = submitted_name, False
                    self.message("[ERROR] %s scheduler failure: %s" % (name, error))
                if not success:
                    failures.append(name)
        if failures:
            self.message("[STAGE FAILED] " + ", ".join(sorted(failures)))
        return failures


def read_revisions() -> Dict[str, str]:
    if not REVISION_LOCK.is_file():
        raise FileNotFoundError(
            "Missing configs/llm_revisions.json. Copy the example and fill every revision with a commit SHA."
        )
    payload = json.loads(REVISION_LOCK.read_text(encoding="utf-8"))
    models = payload.get("models", {})
    datasets = payload.get("datasets", {})
    missing = [model for model in MODELS if model not in models]
    missing_datasets = [
        task.dataset_id for task in TASKS if task.dataset_id not in datasets
    ]
    invalid = [
        value
        for value in list(models.values()) + list(datasets.values())
        if re.fullmatch(r"[0-9a-fA-F]{40}", str(value)) is None
    ]
    if missing or missing_datasets or invalid:
        raise ValueError(
            "Revision lock is incomplete: missing_models=%s missing_datasets=%s invalid_values=%d"
            % (missing, missing_datasets, len(invalid))
        )
    return {str(key): str(value) for key, value in models.items()}


def valid_csv(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    if (
        frame.empty
        or "run_mode" not in frame
        or set(frame["run_mode"].dropna()) != {"full"}
    ):
        return False
    fatal = {
        "accounting_failure",
        "method_failure",
        "numerical_failure",
        "wrapper_failure",
    }
    return "status" not in frame or not set(
        frame["status"].dropna().astype(str)
    ).intersection(fatal)


def valid_protocol_lock() -> bool:
    paths = (
        CANONICAL_DIR / "protocol_lock.json",
        CANONICAL_DIR / "split_manifest.parquet",
        CANONICAL_DIR / "prompt_manifest.parquet",
        CANONICAL_DIR / "expected_run_keys.parquet",
    )
    if not all(path.is_file() for path in paths) or not REVISION_LOCK.is_file():
        return False
    try:
        protocol = json.loads(paths[0].read_text(encoding="utf-8"))
        requested = json.loads(REVISION_LOCK.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        protocol.get("revisions") == requested
        and protocol.get("transformers_commit") == TRANSFORMERS_COMMIT
    )


def valid_candidate_protocol() -> bool:
    paths = (
        CANONICAL_DIR / "protocol_lock.json",
        CANONICAL_DIR / "candidate_manifest.parquet",
        CANONICAL_DIR / "candidate_filter_audit.parquet",
        CANONICAL_DIR / "prompt_manifest.parquet",
    )
    if not all(path.is_file() for path in paths) or not REVISION_LOCK.is_file():
        return False
    try:
        protocol = json.loads(paths[0].read_text(encoding="utf-8"))
        requested = json.loads(REVISION_LOCK.read_text(encoding="utf-8"))
        filter_audit = pd.read_parquet(paths[2])
        candidates = pd.read_parquet(paths[1])
        prompts = pd.read_parquet(paths[3])
    except Exception:
        return False
    key_columns = ["task_id", "item_id"]
    return (
        protocol.get("protocol_version") == PROTOCOL_VERSION
        and protocol.get("revisions") == requested
        and protocol.get("transformers_commit") == TRANSFORMERS_COMMIT
        and not candidates.duplicated(key_columns).any()
        and not prompts.duplicated(key_columns).any()
        and {
            "task_id",
            "source_id",
            "reason",
            "observed_option_count",
            "required_option_count",
            "dataset_revision",
        }.issubset(filter_audit.columns)
    )


def valid_tokenizer_audit() -> bool:
    path = CANONICAL_DIR / "tokenizer_length_audit.parquet"
    split_path = CANONICAL_DIR / "split_manifest.parquet"
    contract_path = CANONICAL_DIR / "split_contract.parquet"
    exclusion_path = CANONICAL_DIR / "split_cluster_exclusion_audit.parquet"
    expected_path = CANONICAL_DIR / "expected_run_keys.parquet"
    candidate_path = CANONICAL_DIR / "candidate_manifest.parquet"
    if not all(
        item.is_file()
        for item in (
            path,
            split_path,
            contract_path,
            exclusion_path,
            expected_path,
            candidate_path,
        )
    ):
        return False
    try:
        frame = pd.read_parquet(path)
        split = pd.read_parquet(split_path)
        contract = pd.read_parquet(contract_path)
        exclusions = pd.read_parquet(exclusion_path)
        expected = pd.read_parquet(expected_path)
        candidates = pd.read_parquet(candidate_path)
        from kernel_isqed.frozen import source_tree_hash
    except Exception:
        return False
    active = split[split["split"].isin(("fit", "validation", "iid_test", "ood_test"))]
    cluster_disjoint = not (active.groupby("item_cluster")["split"].nunique() > 1).any()
    domain_disjoint = True
    shared_in_domain_support = True
    for _, task_rows in active.groupby("task_id"):
        ood = set(task_rows[task_rows["split"] == "ood_test"]["domain"].astype(str))
        iid = set(task_rows[task_rows["split"] != "ood_test"]["domain"].astype(str))
        domain_disjoint = domain_disjoint and not ood.intersection(iid)
        supports = [
            set(task_rows[task_rows["split"] == split]["domain"].astype(str))
            for split in ("fit", "validation", "iid_test")
        ]
        shared_in_domain_support = shared_in_domain_support and all(
            support == supports[0] for support in supports[1:]
        )
    iid_contract = contract[contract["split"] == "iid_test"]
    active_keys = set(
        map(tuple, active[["task_id", "item_id"]].drop_duplicates().to_numpy(),)
    )
    candidate_keys = set(
        map(tuple, candidates[["task_id", "item_id"]].drop_duplicates().to_numpy(),)
    )
    return (
        not frame.empty
        and set(frame["model_id"].astype(str)) == set(MODELS)
        and {"rendered_tokens", "within_limit", "source_tree_hash"}.issubset(frame.columns)
        and set(frame["source_tree_hash"].astype(str)) == {source_tree_hash()}
        and not contract.empty
        and not expected.empty
        and {"MODEL_MAIN", "MODEL_ROBUSTNESS", "PREFLIGHT"}.issubset(
            set(expected["key_family"].astype(str))
        )
        and {
            "task_id",
            "item_cluster",
            "reason",
            "candidate_items",
            "domain_count",
            "gold_count",
            "dataset_revision",
        }.issubset(exclusions.columns)
        and cluster_disjoint
        and domain_disjoint
        and shared_in_domain_support
        and active_keys.issubset(candidate_keys)
        and (
            iid_contract["actual_items"].astype(int)
            == iid_contract["requested_items"].astype(int)
        ).all()
    )


def expected_model_keys(splits: Sequence[str]) -> Set[Tuple[str, str, str]]:
    path = CANONICAL_DIR / "split_manifest.parquet"
    if not path.is_file():
        return set()
    frame = pd.read_parquet(path)
    frame = frame[frame["split"].isin(splits)]
    return set(
        map(tuple, frame[["task_id", "item_id", "split"]].drop_duplicates().to_numpy())
    )


def valid_response_cache(model_id: str, revision: str, splits: Sequence[str]) -> bool:
    path = CANONICAL_DIR / "response_cache" / (model_id.replace("/", "--") + ".parquet")
    if not path.is_file():
        return False
    try:
        frame = pd.read_parquet(path)
        inventory_path = (
            CANONICAL_DIR
            / "preflight"
            / (model_id.replace("/", "--") + "_inventory.json")
        )
        inventory_digest = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    except Exception:
        return False
    required_columns = {
        "model_id",
        "revision",
        "task_id",
        "item_id",
        "split",
        "preflight_inventory_sha256",
        "state_freeze_sha256",
    }
    if not required_columns.issubset(frame.columns):
        return False
    selected = frame[frame["split"].isin(splits)]
    provenance = {
        "prompt_hash",
        "preflight_inventory_sha256",
        "item_cluster",
        "domain",
        "gold",
    }
    if (
        selected.empty
        or set(selected["revision"].astype(str)) != {revision}
        or not provenance.issubset(selected.columns)
        or set(selected["preflight_inventory_sha256"].astype(str)) != {inventory_digest}
    ):
        return False
    if set(splits).intersection({"iid_test", "ood_test"}):
        freeze_path = CANONICAL_DIR / "state_freeze.json"
        if not freeze_path.is_file() or set(
            selected["state_freeze_sha256"].astype(str)
        ) != {hashlib.sha256(freeze_path.read_bytes()).hexdigest()}:
            return False
    if selected.duplicated(["task_id", "item_id", "split"]).any():
        return False
    observed = set(map(tuple, selected[["task_id", "item_id", "split"]].to_numpy()))
    return observed == expected_model_keys(splits)


def valid_preflight(model_id: str, revision: str) -> bool:
    safe_name = model_id.replace("/", "--")
    result_path = CANONICAL_DIR / "preflight" / (safe_name + ".parquet")
    inventory_path = CANONICAL_DIR / "preflight" / (safe_name + "_inventory.json")
    if not result_path.is_file() or not inventory_path.is_file():
        return False
    try:
        frame = pd.read_parquet(result_path)
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        from kernel_isqed.frozen import source_tree_hash
    except Exception:
        return False
    wrapper_valid = True
    if model_id in {"Qwen/Qwen3.5-9B", "google/gemma-4-12B-it"}:
        wrapper_valid = (
            inventory.get("loader_class") == "AutoModelForMultimodalLM"
            and inventory.get("multimodal_wrapper") is True
        )
    if model_id == "mistralai/Ministral-3-8B-Instruct-2512-BF16":
        wrapper_valid = (
            inventory.get("loader_class") == "AutoModelForImageTextToText"
            and inventory.get("multimodal_wrapper") is True
        )
    expected_path = CANONICAL_DIR / "expected_run_keys.parquet"
    if not expected_path.is_file():
        return False
    try:
        expected = pd.read_parquet(expected_path)
    except Exception:
        return False
    expected = expected[
        (expected["key_family"] == "PREFLIGHT")
        & (expected["check"] == "stratified_item")
        & (expected["model_id"] == model_id)
    ]
    observed_keys = set(map(tuple, frame[["task_id", "item_id"]].to_numpy()))
    expected_keys = set(map(tuple, expected[["task_id", "item_id"]].to_numpy()))
    codebooks = inventory.get("task_codebooks", {})
    codebook_modes = {
        codebook.get("codebook_mode") for codebook in codebooks.values()
    }
    expected_codebook_modes = (
        {"trie_sequence_log_probability"}
        if model_id == "mistralai/Mistral-7B-Instruct-v0.3"
        else {"single_token"}
    )
    return (
        len(frame) == 256
        and observed_keys == expected_keys
        and set(frame["status"].astype(str)) == {"completed"}
        and set(frame["length_bin"].astype(int)) == {0, 1, 2, 3}
        and frame["tail_anchor"].astype(bool).any()
        and frame["probability_sum_error"].max() <= PREFLIGHT_NORMALIZATION_TOLERANCE
        and frame["deterministic_max_difference"].max()
        <= PREFLIGHT_DETERMINISM_TOLERANCE
        and frame["centered_logit_max_difference"].max() <= PREFLIGHT_LOGIT_TOLERANCE
        and frame["official_probability_max_difference"].max()
        <= PREFLIGHT_PROBABILITY_TOLERANCE
        and (frame["peak_allocated_bytes"] < frame["vram_limit_bytes"]).all()
        and int(frame["generation_valid"].notna().sum()) == 32
        and inventory.get("requested_revision") == revision
        and inventory.get("weight_dtype") == "bfloat16"
        and inventory.get("transformers_commit") == TRANSFORMERS_COMMIT
        and inventory.get("source_tree_hash") == source_tree_hash()
        and inventory.get("selective_logit_backend")
        == "verified_final_hidden_plus_output_head"
        and inventory.get("code_tokenization")
        in {
            "chat_template_continuation_from_frozen_prefix",
            "tokenizer_text_code_after_frozen_prefix",
        }
        and inventory.get("official_equivalence_items") == 256
        and inventory.get("generation_sanity_items") == 32
        and inventory.get("vram_limit_bytes")
        == int(PREFLIGHT_VRAM_LIMIT_GIB * (1024 ** 3))
        and isinstance(inventory.get("gpu_name"), str)
        and int(inventory.get("gpu_total_memory_bytes", 0)) > 0
        and set(codebooks) == {"mmlu_pro", "hellaswag", "rewardbench"}
        and codebook_modes == expected_codebook_modes
        and inventory.get("preflight_selection")
        == "shared_task_by_four_length_bins_with_per_model_p95_p99_anchors"
        and wrapper_valid
    )


def valid_state_freeze() -> bool:
    freeze_path = CANONICAL_DIR / "state_freeze.json"
    manifest_path = CANONICAL_DIR / "serialized_states.parquet"
    planted_manifest_path = CANONICAL_DIR / "frozen_planted_manifest.json"
    planted_results_path = CANONICAL_DIR / "frozen_planted_results.parquet"
    planted_oracle_path = (
        CANONICAL_DIR.parent / "oracle_truth/frozen_planted_openings.parquet"
    )
    if (
        not freeze_path.is_file()
        or not manifest_path.is_file()
        or not planted_manifest_path.is_file()
        or not planted_results_path.is_file()
        or not planted_oracle_path.is_file()
    ):
        return False
    try:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        manifest = pd.read_parquet(manifest_path)
        from kernel_isqed.frozen import source_tree_hash
        from kernel_isqed.frozen import verify_feature_state_artifacts
        from kernel_isqed.frozen import verify_frozen_inputs
        from kernel_isqed.io import sha256_file

        verify_feature_state_artifacts(manifest)
        verify_frozen_inputs(freeze)
    except Exception:
        return False
    required = {
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
    return (
        freeze.get("protocol") == PROTOCOL_VERSION
        and int(freeze.get("state_count", -1)) == len(manifest)
        and freeze.get("serialized_states_sha256") == sha256_file(manifest_path)
        and required.issubset(set(manifest["state_family"].astype(str)))
        and int(freeze.get("frozen_planted_worlds", 0)) > 0
        and freeze.get("frozen_planted_manifest_sha256")
        == sha256_file(planted_manifest_path)
        and freeze.get("frozen_planted_results_sha256")
        == sha256_file(planted_results_path)
        and freeze.get("frozen_planted_oracle_sha256")
        == sha256_file(planted_oracle_path)
        and freeze.get("test_unlocks") == 0
        and freeze.get("source_tree_hash") == source_tree_hash()
        and freeze.get("requirements_sha256")
        == sha256_file(PROJECT_ROOT / "requirements.txt")
        and freeze.get("python_version_file_sha256")
        == sha256_file(PROJECT_ROOT / ".python-version")
        and freeze.get("pyproject_sha256")
        == sha256_file(PROJECT_ROOT / "pyproject.toml")
    )


def valid_test_unlock() -> bool:
    path = CANONICAL_DIR / "test_unlock.json"
    freeze_path = CANONICAL_DIR / "state_freeze.json"
    if not path.is_file() or not freeze_path.is_file():
        return False
    try:
        unlock = json.loads(path.read_text(encoding="utf-8"))
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        import hashlib

        from kernel_isqed.frozen import source_tree_hash
        from kernel_isqed.frozen import verify_frozen_inputs

        digest = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
        verify_frozen_inputs(freeze)
    except Exception:
        return False
    return (
        unlock.get("unlock_count") == 1
        and unlock.get("state_freeze_sha256") == digest
        and unlock.get("source_tree_hash")
        == freeze.get("source_tree_hash")
        == source_tree_hash()
    )


def valid_robustness_cache(model_id: str, revision: str) -> bool:
    path = (
        CANONICAL_DIR / "robustness_cache" / (model_id.replace("/", "--") + ".parquet")
    )
    keys_path = CANONICAL_DIR / "expected_run_keys.parquet"
    if not path.is_file() or not keys_path.is_file():
        return False
    try:
        observed = pd.read_parquet(path)
        expected = pd.read_parquet(keys_path)
        inventory_path = (
            CANONICAL_DIR
            / "preflight"
            / (model_id.replace("/", "--") + "_inventory.json")
        )
        inventory_digest = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    except Exception:
        return False
    required_columns = {
        "revision",
        "task_id",
        "item_id",
        "split",
        "view_id",
        "prompt_hash",
        "preflight_inventory_sha256",
        "state_freeze_sha256",
    }
    if not required_columns.issubset(observed.columns):
        return False
    expected = expected[
        (expected["key_family"] == "MODEL_ROBUSTNESS")
        & (expected["model_id"] == model_id)
    ]
    if set(observed["revision"].astype(str)) != {revision} or set(
        observed["preflight_inventory_sha256"].astype(str)
    ) != {inventory_digest}:
        return False
    freeze_path = CANONICAL_DIR / "state_freeze.json"
    if not freeze_path.is_file() or set(
        observed["state_freeze_sha256"].astype(str)
    ) != {hashlib.sha256(freeze_path.read_bytes()).hexdigest()}:
        return False
    columns = ["task_id", "item_id", "split", "view_id"]
    if observed.duplicated(columns).any():
        return False
    return set(map(tuple, observed[columns].to_numpy())) == set(
        map(tuple, expected[columns].to_numpy())
    )


def matched_responses_complete() -> bool:
    path = CANONICAL_DIR / "matched_responses.parquet"
    expected_path = CANONICAL_DIR / "expected_run_keys.parquet"
    if not path.is_file() or not expected_path.is_file():
        return False
    try:
        observed = pd.read_parquet(path)
        expected = pd.read_parquet(expected_path)
    except Exception:
        return False
    expected = expected[expected["key_family"] == "MODEL_MAIN"]
    columns = ["model_id", "task_id", "item_id", "split"]
    if observed.duplicated(columns).any():
        return False
    return set(map(tuple, observed[columns].to_numpy())) == set(
        map(tuple, expected[columns].to_numpy())
    )


def model_jobs(
    script: str,
    revisions: Dict[str, str],
    validator_factory: Callable[[str, str], bool],
    extra_arguments: Sequence[str] = (),
    model_ids: Sequence[str] = MODELS,
) -> List[Job]:
    jobs = []
    for model_id in model_ids:
        revision = revisions[model_id]
        command = [script, "--model-id", model_id, "--revision", revision] + list(
            extra_arguments
        )
        jobs.append(
            Job(
                name=Path(script).stem + "_" + model_id.replace("/", "--"),
                command=command,
                validator=lambda model_id=model_id, revision=revision: validator_factory(
                    model_id, revision
                ),
                minimum_free_gb=MODEL_JOB_MINIMUM_FREE_GIB,
            )
        )
    return jobs


def run_checked(
    command: Sequence[str], name: str, validator: Callable[[], bool], dry_run: bool
) -> None:
    if validator():
        print("[SKIP] %s already complete." % name, flush=True)
        return
    full_command = [sys.executable] + list(command)
    print("[RUN] " + " ".join(full_command), flush=True)
    if dry_run:
        return
    completed = subprocess.run(full_command, cwd=PROJECT_ROOT)
    if completed.returncode != 0 or not validator():
        raise RuntimeError("%s failed or did not produce its required output" % name)


def gpu_stage_ready(stage: str, dry_run: bool) -> bool:
    if dry_run:
        return True
    try:
        gpus = GPUManager.query()
    except RuntimeError as error:
        print("[BLOCKED] %s requires a visible NVIDIA GPU: %s" % (stage, error))
        return False
    largest = max(gpu.total_gb for gpu in gpus)
    if largest < MODEL_JOB_MINIMUM_FREE_GIB:
        print(
            "[BLOCKED] %s requires a %.1f GiB GPU; largest visible GPU has %.1f GiB."
            % (stage, MODEL_JOB_MINIMUM_FREE_GIB, largest)
        )
        return False
    summary = ", ".join(
        "GPU %d %.1f/%.1f GiB free" % (gpu.index, gpu.free_gb, gpu.total_gb)
        for gpu in gpus
    )
    print("[GPU] " + summary, flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and resume the frozen Target-Imprint LLM experiment pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "core",
            "prepare",
            "preflight",
            "fit",
            "freeze",
            "unlock",
            "test",
            "robustness",
            "experiments",
            "artifacts",
        ),
        default="all",
    )
    parser.add_argument(
        "--mode",
        choices=("formal", "exploration"),
        default="formal",
        help="exploration mode limits --stage all/core to preparation, preflight, and fit",
    )
    parser.add_argument("--experiments", type=int, nargs="*", choices=range(1, 15))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--freeze-jobs", type=int, default=1)
    parser.add_argument("--freeze-task-workers", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for name in ("workers", "freeze_jobs", "freeze_task_workers"):
        if getattr(args, name) < 1:
            parser.error("--%s must be at least one" % name.replace("_", "-"))
    if args.mode == "exploration":
        os.environ["KERNEL_ISQED_EXPLORATION"] = "1"

    revisions = read_revisions()
    runner = TargetImprintRunner(
        revisions,
        max_retries=args.max_retries,
        workers=args.workers,
        poll_seconds=args.poll_seconds,
        force=args.force,
        dry_run=args.dry_run,
    )

    stages = (
        "prepare",
        "preflight",
        "fit",
        "freeze",
        "unlock",
        "test",
        "robustness",
        "experiments",
        "artifacts",
    )
    core_stages = stages[:7]
    exploration_stages = stages[:3]
    if args.stage == "all":
        selected_stages = exploration_stages if args.mode == "exploration" else stages
    elif args.stage == "core":
        selected_stages = (
            exploration_stages if args.mode == "exploration" else core_stages
        )
    else:
        selected_stages = (args.stage,)

    print(
        "[PLAN] mode=%s stages=%s workers=%d freeze_jobs=%d freeze_task_workers=%d"
        % (
            args.mode,
            ",".join(selected_stages),
            args.workers,
            args.freeze_jobs,
            args.freeze_task_workers,
        ),
        flush=True,
    )

    if "prepare" in selected_stages:
        run_checked(
            ["scripts/freeze_llm_protocol.py", "--revision-lock", str(REVISION_LOCK)],
            "freeze_llm_protocol",
            valid_candidate_protocol,
            args.dry_run,
        )
        run_checked(
            [
                "scripts/audit_tokenizer_intersection.py",
                "--revision-lock",
                str(REVISION_LOCK),
            ],
            "tokenizer_intersection",
            valid_tokenizer_audit,
            args.dry_run,
        )
        if not args.dry_run and not valid_protocol_lock():
            raise RuntimeError(
                "tokenizer intersection did not produce a complete frozen split"
            )

    if "preflight" in selected_stages:
        if not gpu_stage_ready("preflight", args.dry_run):
            return 2
        jobs = model_jobs("scripts/preflight_llm.py", revisions, valid_preflight)
        if runner.run_batch(jobs):
            return 1

    if "fit" in selected_stages:
        if not gpu_stage_ready("fit", args.dry_run):
            return 2

        def fit_cache_valid(model: str, revision: str) -> bool:
            return valid_response_cache(model, revision, ("fit", "validation"))

        jobs = model_jobs(
            "scripts/cache_llm_responses.py",
            revisions,
            fit_cache_valid,
            ("--stage", "fit_validation"),
        )
        if runner.run_batch(jobs):
            return 1
        run_checked(
            ["scripts/build_matched_responses.py", "--splits", "fit", "validation"],
            "build_fit_validation_responses",
            lambda: (CANONICAL_DIR / "matched_responses.parquet").is_file()
            and set(
                pd.read_parquet(CANONICAL_DIR / "matched_responses.parquet")["split"]
            )
            == {"fit", "validation"},
            args.dry_run,
        )

    if "freeze" in selected_stages:
        run_checked(
            [
                "scripts/freeze_adapter_states.py",
                "--jobs",
                str(args.freeze_jobs),
                "--task-workers",
                str(args.freeze_task_workers),
            ],
            "freeze_adapter_states",
            valid_state_freeze,
            args.dry_run,
        )

    if "unlock" in selected_stages:
        run_checked(
            ["scripts/unlock_test_once.py"],
            "unlock_test_once",
            valid_test_unlock,
            args.dry_run,
        )

    if "test" in selected_stages:
        if not gpu_stage_ready("test", args.dry_run):
            return 2
        if not args.dry_run and not valid_test_unlock():
            raise RuntimeError(
                "run --stage unlock once before requesting sealed test responses"
            )

        def test_cache_valid(model: str, revision: str) -> bool:
            return valid_response_cache(model, revision, ("iid_test", "ood_test"))

        jobs = model_jobs(
            "scripts/cache_llm_responses.py",
            revisions,
            test_cache_valid,
            ("--stage", "test"),
        )
        if runner.run_batch(jobs):
            return 1
        run_checked(
            ["scripts/build_matched_responses.py"],
            "build_full_matched_responses",
            matched_responses_complete,
            args.dry_run,
        )

    if "robustness" in selected_stages:
        if not gpu_stage_ready("robustness", args.dry_run):
            return 2
        jobs = model_jobs(
            "scripts/cache_llm_robustness.py",
            revisions,
            valid_robustness_cache,
            model_ids=REFERENCE_MODELS,
        )
        if runner.run_batch(jobs):
            return 1

    if "experiments" in selected_stages:
        if not args.dry_run and not matched_responses_complete():
            raise RuntimeError(
                "full real-model matched_responses.parquet is incomplete; refusing to run formal experiments"
            )
        chosen = set(args.experiments or range(1, 15))
        jobs = [
            Job(
                name="exp%d" % number,
                command=[script],
                validator=lambda output=output: valid_csv(RESULTS_DIR / output),
                minimum_free_gb=2.0,
                needs_gpu=False,
            )
            for number, script, output in EXPERIMENTS
            if number in chosen
        ]
        failures = runner.run_batch(jobs)
        if failures:
            return 1

    if "artifacts" in selected_stages:
        run_checked(
            ["scripts/build_artifacts.py"],
            "build_artifacts",
            lambda: (PROJECT_ROOT / "artifact_manifest.json").is_file()
            and json.loads(
                (PROJECT_ROOT / "artifact_manifest.json").read_text(encoding="utf-8")
            ).get("count")
            == 21,
            args.dry_run,
        )

    print("[DONE] Requested Target-Imprint stages completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
