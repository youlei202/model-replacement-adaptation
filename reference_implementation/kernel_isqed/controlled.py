from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict
import pandas as pd

from .io import CANONICAL_DIR


P = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E08"
    "8A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B"
    "302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E"
    "9A63A3620FFFFFFFFFFFFFFFF",
    16,
)
Q = (P - 1) // 2
G = 4


def _hash_to_group(label: str) -> int:
    value = int(hashlib.sha512(label.encode("utf-8")).hexdigest(), 16) % P
    element = pow(value, 2, P)
    return element if element not in (0, 1) else pow(5, 2, P)


H = _hash_to_group("Target-Imprint Pedersen h v1")


@dataclass(frozen=True)
class Commitment:
    commitment_id: str
    value: int
    blinding: int


@dataclass(frozen=True)
class WorldOpening:
    commitment_scope: str
    seed: int
    commitment: Commitment


def pedersen_commit(message: int) -> Commitment:
    blinding = secrets.randbelow(Q)
    value = (pow(G, int(message) % Q, P) * pow(H, blinding, P)) % P
    encoded = value.to_bytes((P.bit_length() + 7) // 8, "big")
    commitment_id = hashlib.sha256(encoded).hexdigest()
    return Commitment(commitment_id, value, blinding)


def verify_commitment(message: int, commitment: Commitment) -> bool:
    expected = (pow(G, int(message) % Q, P) * pow(H, commitment.blinding, P)) % P
    return expected == commitment.value


def world_opening_path() -> Path:
    return CANONICAL_DIR.parent / "oracle_truth/expected_world_openings.parquet"


@lru_cache(maxsize=2)
def _registered_worlds(path_text: str) -> Dict[str, WorldOpening]:
    path = Path(path_text)
    if not path.is_file():
        return {}
    frame = pd.read_parquet(path)
    worlds = {}
    for row in frame.itertuples(index=False):
        commitment = Commitment(str(row.commitment_id), int(row.commitment_value), int(row.blinding),)
        opening = WorldOpening(str(row.commitment_scope), int(str(row.seed_hex), 16), commitment,)
        worlds[opening.commitment_scope] = opening
    return worlds


@lru_cache(maxsize=None)
def load_world_opening(commitment_scope: str,) -> WorldOpening:
    path = world_opening_path()
    opening = _registered_worlds(str(path)).get(commitment_scope)
    if opening is not None:
        if not verify_commitment(opening.seed, opening.commitment):
            raise RuntimeError("registered world commitment does not verify: %s" % commitment_scope)
        return opening
    raise FileNotFoundError(
        "registered controlled-world opening is missing for %s; run freeze_llm_protocol.py first" % commitment_scope
    )
