"""Reproduce the bounded fit-only diagnostics after the first AURUM sprint."""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .codec import decode
from .data import RichDevelopmentCache
from .diagnostics import feature_conditioning, per_query_basis_oracle, tiny_fit_diagnostic
from .models import AnchoredResidual, FrozenAnchor
from .train import ArrayData

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=int, nargs="+", default=[4, 3])
    parser.add_argument("--peers", type=int, nargs=2, default=[1, 2])
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--n-examples", type=int, default=32)
    parser.add_argument("--map-steps", type=int, default=300)
    parser.add_argument("--map-lrs", type=float, nargs="+", default=[.001, .01])
    parser.add_argument("--oracle-steps", type=int, default=400)
    parser.add_argument("--oracle-lr", type=float, default=.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    folder = Path(args.output).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError("Preserve existing results: choose a new diagnostic output directory")
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    pair = tuple(args.peers)
    cache = RichDevelopmentCache()
    context = ROOT / "runs/aurum/public_contexts" / f"pair_{pair[0]}_{pair[1]}" / f"rank_{args.rank}"
    with np.load(context / "features.npz") as bank:
        features = bank["fit"].copy()
    basis = torch.from_numpy(np.array(cache.basis(pair, args.rank), copy=True)).to(device)
    peers = tuple(cache.logits("fit", p) for p in pair)
    start = time.time()
    sources = {name: sha(ROOT / "aurum" / name) for name in
               ("diagnose.py", "diagnostics.py", "models.py", "losses.py", "codec.py", "train.py", "data.py")}
    save(folder / "run_spec.json", {"arguments": vars(args), "host": socket.gethostname(),
         "torch_version": torch.__version__, "cuda_device": torch.cuda.get_device_name(device),
         "source_sha256": sources, "started_unix": start, "target_validation_logits_opened": False,
         "basis_context_digest": sha(context / "context.json"),
         "role": "Fit-only memorization and query-specific capacity diagnostics; not deployable candidates"})
    rows = []
    for target in args.targets:
        cell = ROOT / "runs/aurum/development/cells" / f"t{target}_p{pair[0]}-{pair[1]}"
        ledger = pd.read_csv(cell / "candidate_ledger.csv")
        calibrated = ledger[ledger.cheap & ledger.label.str.startswith("calibration_")]
        selected = calibrated.sort_values(["inner_stop_tv", "actual_bytes", "candidate_id"]).iloc[0]
        state_path = cell / selected.state_path
        state = decode(state_path.read_bytes())
        kind = "probability" if state["family"].endswith("probability") else "logit"
        anchor = FrozenAnchor(torch.as_tensor(state["arrays"][0], device=device), kind)
        train_ids = np.asarray(json.loads((cell / "data_roles.json").read_text())["train_indices"])
        data = ArrayData(peers, features, cache.logits("fit", target), train_ids, role="inner_train")
        provenance = {"target": target, "pair": list(pair), "basis_rank": args.rank,
                      "anchor_label": selected.label, "anchor_candidate_id": selected.candidate_id,
                      "anchor_selected_on": "inner_stop_TV_among_probability_trained_calibration_candidates",
                      "anchor_state_sha256": sha(state_path),
                      "fit_input_hashes": {str(i): cache.metadata("fit", i)["sha256"] for i in (*pair, target)}}
        save(folder / f"t{target}_feature_conditioning.json", {**provenance,
             "inner_training_features": feature_conditioning(features, train_ids)})
        jobs = [(f"tiny_map_lr{lr:g}", lr) for lr in args.map_lrs] + [("per_query_oracle", args.oracle_lr)]
        for label, lr in jobs:
            def progress(row):
                if row["step"] % 100 == 0:
                    print(json.dumps({"target": target, "diagnostic": label, **row}), flush=True)
            if label.startswith("tiny"):
                model = AnchoredResidual(anchor, basis, features.shape[1])
                result = tiny_fit_diagnostic(model, data, n_examples=args.n_examples,
                    steps=args.map_steps, lr=lr, seed=args.seed, callback=progress)
            else:
                result = per_query_basis_oracle(anchor, basis, data, n_examples=args.n_examples,
                    steps=args.oracle_steps, lr=lr, seed=args.seed, callback=progress)
            result.update(provenance)
            save(folder / f"t{target}_{label}.json", result)
            row = {"target": target, "pair": ",".join(map(str, pair)), "basis_rank": args.rank,
                   "diagnostic": label, "n_examples": result["n_examples"], "lr": lr,
                   "before_tv": result["before"]["tv"], "best_tv": result["best"]["tv"],
                   "before_kl": result["before"]["kl"], "best_kl": result["best"]["kl"],
                   "absolute_fit_tv_gain": result["absolute_fit_tv_gain"], "best_step": result["best_step"],
                   "completed_steps": result["completed_steps"], "seconds": result["seconds"],
                   "diagnostic_only": True, "independent_confirmation": False}
            rows.append(row)
            pd.DataFrame(rows).to_csv(folder / "DIAGNOSTIC_SUMMARY.csv", index=False)
            print(json.dumps({"completed": row}), flush=True)
    assert all(sha(ROOT / "aurum" / name) == digest for name, digest in sources.items())
    save(folder / "complete.json", {"completed": True, "diagnostic_jobs": len(rows),
         "elapsed_seconds": time.time() - start, "summary_sha256": sha(folder / "DIAGNOSTIC_SUMMARY.csv"),
         "max_GPU_allocated_bytes": torch.cuda.max_memory_allocated(device), "source_sha256": sources})


if __name__ == "__main__":
    main()
