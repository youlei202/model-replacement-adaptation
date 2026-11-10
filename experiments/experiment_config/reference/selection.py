"""Validation-only joint/fixed policies. All inputs refer to one target/task/fit.

The production runner must fit and encode candidates first. This module never
accepts test losses and never treats native FP64 as an irreducible optimum.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    target_id: str
    peer_ids: tuple[str, ...]
    family: str
    actual_bits: int
    validation_loss: float
    native_full: bool = False

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.target_id or not self.peer_ids:
            raise ValueError("IDs and peer family must be nonempty")
        if len(set(self.peer_ids)) != len(self.peer_ids):
            raise ValueError("Duplicate peer in an interface")
        if self.target_id in self.peer_ids:
            raise ValueError("Target cannot be a deployed peer")
        if tuple(sorted(self.peer_ids)) != self.peer_ids:
            raise ValueError("Peer IDs must be in canonical sorted order")
        if self.actual_bits < 0 or not isinstance(self.actual_bits, int):
            raise ValueError("Actual bits must be a nonnegative integer")
        if not isfinite(self.validation_loss) or self.validation_loss < 0:
            raise ValueError("Invalid validation loss")

    @classmethod
    def from_validation_record(cls, record: Mapping) -> "Candidate":
        forbidden = [k for k in record if 'test' in k.lower() or 'oracle' in k.lower()]
        if forbidden:
            raise ValueError(f"Selection input contains forbidden fields: {forbidden}")
        return cls(
            str(record['candidate_id']), str(record['target_id']),
            tuple(sorted(map(str, record['peer_ids']))), str(record['family']),
            int(record['actual_bits']), float(record['validation_loss']),
            bool(record.get('native_full', False)),
        )


def peer_family(registry: Sequence[str], target: str, q: int) -> list[tuple[str, ...]]:
    if len(set(registry)) != len(registry) or target not in registry:
        raise ValueError("Registry must be unique and contain target")
    peers = sorted(p for p in registry if p != target)
    if not 1 <= q <= len(peers):
        raise ValueError("Invalid call count")
    return list(combinations(peers, q))


def _tie(c: Candidate) -> tuple:
    return (c.validation_loss, c.actual_bits, c.family, c.peer_ids, c.candidate_id)


def _validate(candidates: Iterable[Candidate]) -> list[Candidate]:
    cs = list(candidates)
    if not cs or len({c.target_id for c in cs}) != 1:
        raise ValueError("Policies must operate within one nonempty target/task/fit")
    if len({len(c.peer_ids) for c in cs}) != 1:
        raise ValueError("Use matched-q candidates for this policy comparison")
    ids = [c.candidate_id for c in cs]
    if len(ids) != len(set(ids)):
        raise ValueError("Deduplicate candidate states before policy selection")
    return cs


def best_at(candidates: Sequence[Candidate], budget: int | None,
            peers: tuple[str, ...] | None = None) -> Candidate:
    if budget is not None and budget < 0:
        raise ValueError("Use None, not a negative cap, for unrestricted state")
    feasible = [c for c in candidates if (budget is None or c.actual_bits <= budget)
                and (peers is None or c.peer_ids == peers)]
    if not feasible:
        raise ValueError(f"Infeasible interface/cap: {peers}, {budget}")
    return min(feasible, key=_tie)


def select_policies(candidates: Iterable[Candidate], budgets: Sequence[int],
                    weights: Sequence[float] | None = None) -> dict[str, dict[int, Candidate]]:
    cs = _validate(candidates)
    bs = tuple(budgets)
    if not bs or len(set(bs)) != len(bs) or any(b < 0 for b in bs):
        raise ValueError("Unique nonnegative primary budgets are required")
    ws = tuple(weights) if weights is not None else (1 / len(bs),) * len(bs)
    if len(ws) != len(bs) or any(w < 0 for w in ws) or abs(sum(ws) - 1) > 1e-10:
        raise ValueError("Budget weights must be nonnegative and sum to one")
    pairs = sorted({c.peer_ids for c in cs})
    # A pair that is infeasible is not silently removed from fixed comparisons.
    by = {(p, b): best_at(cs, b, p) for p in pairs for b in bs}
    dynamic = {b: best_at(cs, b) for b in bs}
    native = [c for c in cs if c.native_full]
    if {c.peer_ids for c in native} != set(pairs):
        raise ValueError("Every interface needs its native-full reference")
    fixed_native = min(native, key=_tie).peer_ids
    fixed_unbounded = best_at(cs, None).peer_ids
    fixed_mean = min(pairs, key=lambda p: (
        sum(w * by[p, b].validation_loss for b, w in zip(bs, ws)), p))
    fixed_minimax = min(pairs, key=lambda p: (
        max(by[p, b].validation_loss - dynamic[b].validation_loss for b in bs),
        sum(w * by[p, b].validation_loss for b, w in zip(bs, ws)), p))
    out = {'budget_aware': dynamic}
    for name, p in [('fixed_native_full', fixed_native),
                    ('fixed_unbounded_best', fixed_unbounded),
                    ('fixed_mean_budget', fixed_mean),
                    ('fixed_minimax_validation_regret', fixed_minimax)]:
        out[name] = {b: by[p, b] for b in bs}
    nnls = [c for c in cs if c.family == 'nnls']
    if nnls:
        out['budget_aware_nnls'] = {b: best_at(nnls, b) for b in bs}
    return out
