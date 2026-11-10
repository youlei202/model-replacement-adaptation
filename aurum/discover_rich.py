"""Shared stronger-control catalog after AURUM's anchor/loss diagnostic.

Every pair receives the same bounded architecture/optimizer menu. Screening and
continuation decisions use inner stopping TV; every decoded endpoint is retained
in the catalog, including ties, zero corrections and worse candidates.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .codec import decode
from .data import fit_legacy_weights
from .models import AnchoredMLP, AnchoredResidual, AnchoredResponseGate, FrozenAnchor
from .robust_calibration import optimize_coefficients
from .session import DevelopmentSession
from .sprint import ROOT, dump, sha

CODECS = ("fp32", "fp16", "int8", "int4")
RECIPES = (("adam_1e3", .001, 0.), ("adam_3e4_shrink", .0003, .0001))
SCIENTIFIC_SOURCES = ("discover_rich.py", "session.py", "robust_calibration.py", "sprint.py",
                      "data.py", "models.py", "losses.py", "train.py", "codec.py", "inference.py")


def architecture_menu():
    """The common 15-architecture menu, before two optimizer recipes each."""
    specs = [{"family": "linear", "context": f"logit{rank}", "output_rank": rank}
             for rank in (16, 32, 64, 128)]
    specs += [{"family": "factored", "context": "logit128", "output_rank": 128, "factor_rank": k}
              for k in (1, 2, 4, 8, 16)]
    specs += [{"family": "mlp", "context": "logit128", "output_rank": 128, "hidden": h}
              for h in (16, 32, 64)]
    specs += [{"family": "gate", "context": context, "output_rank": 0}
              for context in ("stats", "logit32", "logit128")]
    return specs


def architecture_name(spec):
    name = spec["family"] + "_" + spec["context"]
    if "factor_rank" in spec:
        name += "_k" + str(spec["factor_rank"])
    if "hidden" in spec:
        name += "_h" + str(spec["hidden"])
    return name


def make_model(session, spec, anchor):
    basis, features, _ = session.context(spec["context"])
    dimension = features["fit"].shape[1]
    # Only the tiny target-dependent anchor is copied, never the public basis.
    anchor = copy.deepcopy(anchor)
    if spec["family"] in ("linear", "factored"):
        return AnchoredResidual(anchor, basis, dimension, factor_rank=spec.get("factor_rank"), seed=session.seed)
    if spec["family"] == "mlp":
        return AnchoredMLP(anchor, basis, dimension, hidden=spec["hidden"], seed=session.seed)
    if spec["family"] == "gate":
        return AnchoredResponseGate(anchor, dimension)
    raise ValueError("Unknown architecture")


def load_decoded_parameters(model, state):
    """Continue from the actually scored state, including quantized anchor."""
    reference = model.anchor.weights
    a = [torch.as_tensor(x, device=reference.device, dtype=reference.dtype) for x in state["arrays"]]
    with torch.no_grad():
        model.anchor.weights.copy_(FrozenAnchor(a[0], model.anchor.kind).weights)
        if isinstance(model, AnchoredResidual):
            model.bias.copy_(a[1])
            if model.factor_rank is None:
                model.linear.copy_(a[2])
            else:
                model.down.copy_(a[2]); model.up.copy_(a[3])
        elif isinstance(model, AnchoredMLP):
            model.hidden.weight.copy_(a[1]); model.hidden.bias.copy_(a[2])
            model.output.weight.copy_(a[3]); model.output.bias.copy_(a[4])
        elif isinstance(model, AnchoredResponseGate):
            model.gate.weight.copy_(a[1]); model.gate.bias.copy_(a[2]); model.strength = float(a[3][0])
        else:
            raise ValueError("Unknown continuation architecture")
    return model


def inner_selection(rows):
    """No outer loss enters any continuation or anchor-selection decision."""
    return min(rows, key=lambda row: (row["inner_stop_tv"], row["actual_bytes"], row["candidate_id"]))


def offer_codecs(session, model, label, context="none", extra=None):
    return [session.offer(model, label, context=context, codec=codec, extra=extra) for codec in CODECS]


def strengthened_cheap_menu(session, *, maxiter=60):
    train, _, _ = session.datasets("none")
    weights = fit_legacy_weights(tuple(session.cache.logits("fit", p) for p in session.pair),
                                 session.cache.logits("fit", session.target), session.train_ids,
                                 device=str(session.device), batch_size=32)
    dump(session.folder / "legacy_inner_weights.json", {key: value.tolist() if isinstance(value, np.ndarray) else value
                                                       for key, value in weights.items()})
    rows = []
    starts = [("legacy_nnls", weights["nnls"], "logit"), ("legacy_convex", weights["convex"], "logit"),
              ("equal_logit", np.array([.5, .5], np.float32), "logit"),
              ("equal_probability", np.array([.5, .5], np.float32), "probability")]
    starts += [(f"direct_peer_{j}", np.eye(2, dtype=np.float32)[j], "logit") for j in range(2)]
    for label, coefficients, kind in starts:
        model = FrozenAnchor(torch.as_tensor(coefficients, device=session.device), kind)
        rows += offer_codecs(session, model, label, extra={"stage": "cheap", "objective": "legacy_or_direct",
                                                         "anchor_training_rows": len(train)})

    def optimize(label, initial, kind, loss, single=None):
        model, info = optimize_coefficients(train, initial, kind=kind, loss=loss, single=single,
                                            maxiter=maxiter, device=str(session.device), batch_size=128)
        dump(session.folder / "training" / (label + ".json"), info)
        rows.extend(offer_codecs(session, model, label, extra={
            "stage": "cheap", "objective": loss, "calibration_kind": kind,
            "calibration_single_peer_position": single, "optimizer_success": info["success"],
            "optimizer_message": info["message"], "training_seconds": info["seconds"],
            "anchor_training_rows": len(train), "optimizer_parameterization": "direct_bounded_coefficients"}))
        return np.asarray(info["weights"], dtype=np.float32)

    logit_kl = optimize("bounded_logit_KL", weights["nnls"], "logit", "kl")
    optimize("bounded_logit_TV_from_KL", logit_kl, "logit", "tv")
    optimize("bounded_logit_TV_from_convex", weights["convex"], "logit", "tv")
    optimize("bounded_logit_TVKL_from_KL", logit_kl, "logit", "tv_kl")
    probability_kl = optimize("bounded_probability_KL", np.array([.5, .5]), "probability", "kl")
    optimize("bounded_probability_TV_from_KL", probability_kl, "probability", "tv")
    optimize("bounded_probability_TVKL_from_KL", probability_kl, "probability", "tv_kl")
    for peer in range(2):
        initial = np.eye(2, dtype=np.float32)[peer]
        single_kl = optimize(f"bounded_single{peer}_KL", initial, "logit", "kl", single=peer)
        optimize(f"bounded_single{peer}_TV_from_KL", single_kl, "logit", "tv", single=peer)
        optimize(f"bounded_single{peer}_TVKL_from_KL", single_kl, "logit", "tv_kl", single=peer)
    selected = inner_selection(rows)
    state = decode(Path(selected["state_path"]).read_bytes())
    kind = "probability" if state["family"].endswith("probability") else "logit"
    anchor = FrozenAnchor(torch.as_tensor(state["arrays"][0], device=session.device), kind)
    dump(session.folder / "anchor_choice.json", {"rule": "minimum decoded inner-stop TV, then bytes, then state digest",
         "candidate": selected, "anchor_weights": anchor.weights.detach().cpu().tolist(),
         "anchor_kind": kind, "outer_validation_not_used_for_choice": True,
         "all_cheap_candidates_remain_available_at_original_sizes": True})
    return anchor, selected


def run(args):
    start = time.time()
    sources = {name: sha(ROOT / "aurum" / name) for name in SCIENTIFIC_SOURCES}
    session = DevelopmentSession(args.output, args.target, args.peers, device=args.device,
                                 seed=args.seed, fold=args.inner_fold)
    specification = {"arguments": vars(args), "source_sha256": sources,
                     "architecture_menu": architecture_menu(), "optimizer_recipes": RECIPES,
                     "codecs": CODECS, "continuation_rule": "one minimum-inner-stop-TV screened candidate per architecture family",
                     "all_pairs_receive_same_search_allocation": True,
                     "scope": "Historical rich fit/validation exploratory development; no independent confirmation",
                     "checkpoint_selection": "inner_stop_TV", "started_unix": start,
                     "original_RBV2_fallbacks": "Separate immutable catalog integration by campaign runner"}
    dump(session.folder / "run_spec.json", specification)
    try:
        anchor, anchor_row = strengthened_cheap_menu(session, maxiter=args.calibration_maxiter)
        screen_rows = []
        screen_specs = {}
        for spec in architecture_menu():
            for recipe_name, lr, weight_decay in RECIPES:
                label = "screen_" + architecture_name(spec) + "_" + recipe_name
                model = make_model(session, spec, anchor)
                train, _, _ = session.datasets(spec["context"])
                peers, features, _ = train.batch(np.arange(min(8, len(train))), session.device)
                with torch.no_grad():
                    assert torch.equal(model(peers, features), anchor(peers)), "Initial correction must reproduce the anchor"
                info = session.optimize(model, label, spec["context"], loss="kl", epochs=args.screen_epochs,
                                        lr=lr, weight_decay=weight_decay, batch_size=args.batch_size,
                                        patience=args.screen_patience)
                extra = {"stage": "screen", "architecture": spec["family"], "architecture_spec": json.dumps(spec, sort_keys=True),
                         "objective": "kl", "recipe": recipe_name, "learning_rate": lr, "weight_decay": weight_decay,
                         "anchor_candidate_id": anchor_row["candidate_id"], "chosen_epoch": info["chosen_epoch"],
                         "completed_epochs": info["completed_epochs"], "training_seconds": info["seconds"],
                         "first_step_gradient_norms": json.dumps(info["first_step_gradient_norms"], sort_keys=True)}
                screen_rows += offer_codecs(session, model, label, spec["context"], extra)
                screen_specs[label] = {"spec": spec, "recipe": recipe_name, "lr": lr, "weight_decay": weight_decay,
                                       "screen_completed_epochs": info["completed_epochs"]}
        selections = {family: inner_selection([row for row in screen_rows if row["architecture"] == family])
                      for family in ("linear", "factored", "mlp", "gate")}
        dump(session.folder / "continuation_choices.json", {"rule": specification["continuation_rule"],
             "selected_on": "inner_stop_TV_only", "choices": selections})
        for family, selected in selections.items():
            entry = screen_specs[selected["label"]]
            spec = entry["spec"]
            model = make_model(session, spec, anchor)
            load_decoded_parameters(model, decode(Path(selected["state_path"]).read_bytes()))
            extra_epochs = max(1, args.total_epochs - entry["screen_completed_epochs"])
            label = "extend_" + selected["label"][7:]
            info = session.optimize(model, label, spec["context"], loss="kl", epochs=extra_epochs,
                                    lr=entry["lr"], weight_decay=entry["weight_decay"], batch_size=args.batch_size,
                                    patience=args.extend_patience)
            extra = {"stage": "extend", "architecture": family, "architecture_spec": json.dumps(spec, sort_keys=True),
                     "objective": "kl", "recipe": entry["recipe"], "learning_rate": entry["lr"],
                     "weight_decay": entry["weight_decay"], "continued_from_candidate_id": selected["candidate_id"],
                     "continued_from_codec": selected["codec"], "chosen_epoch": info["chosen_epoch"],
                     "completed_epochs": info["completed_epochs"], "training_seconds": info["seconds"]}
            offer_codecs(session, model, label, spec["context"], extra)
            for loss in ("tv", "tv_kl"):
                # Copy only target parameters into a fresh model sharing public context.
                refined = make_model(session, spec, anchor)
                refined.load_state_dict(model.state_dict())
                refinement_label = label + "_" + loss
                refinement = session.optimize(refined, refinement_label, spec["context"], loss=loss,
                    epochs=args.tv_epochs, lr=.0001, weight_decay=entry["weight_decay"],
                    batch_size=args.batch_size, patience=args.extend_patience)
                offer_codecs(session, refined, refinement_label, spec["context"], {
                    **extra, "stage": "refine", "objective": "kl_then_" + loss,
                    "learning_rate": .0001, "chosen_epoch": refinement["chosen_epoch"],
                    "completed_epochs": refinement["completed_epochs"], "training_seconds": refinement["seconds"]})
        assert all(sha(ROOT / "aurum" / name) == digest for name, digest in sources.items()), "Scientific source changed during catalog construction"
        session.finish(specification)
    except Exception as error:
        dump(session.folder / "failure.json", {"error": repr(error), "candidate_rows": len(session.rows),
             "outputs_retained": True, "elapsed_seconds": time.time() - start, "source_sha256": sources})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--peers", type=int, nargs=2, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--inner-fold", type=int, default=0)
    parser.add_argument("--screen-epochs", type=int, default=30)
    parser.add_argument("--screen-patience", type=int, default=10)
    parser.add_argument("--total-epochs", type=int, default=100)
    parser.add_argument("--extend-patience", type=int, default=20)
    parser.add_argument("--tv-epochs", type=int, default=30)
    parser.add_argument("--calibration-maxiter", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.target not in (3, 4) or tuple(args.peers) not in ((0, 1), (0, 2), (1, 2)):
        parser.error("The registered rich development population has targets3/4 and pairs01/02/12")
    if min(args.screen_epochs, args.total_epochs, args.tv_epochs, args.calibration_maxiter, args.batch_size) < 1:
        parser.error("Optimization budgets must be positive")
    run(args)


if __name__ == "__main__":
    main()
