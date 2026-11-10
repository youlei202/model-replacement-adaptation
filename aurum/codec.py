"""Literal AUR1 deployment codec. Counts every byte; never use pickle.

Production must keep legacy RBV2 replay and register new families deliberately.
Public feature/basis context is supplied separately, with a recorded digest.
"""
from __future__ import annotations
import math
import struct
import numpy as np

HEADER = struct.Struct("<4sBBBBIIIIII")  # 32 bytes
ARRAY = struct.Struct("<IIIIf")          # 20 bytes per array
MAGIC = b"AUR1"
FAMILIES = {"anchor_logit": 0, "anchor_probability": 1,
            "linear_logit": 2, "linear_probability": 3,
            "factored_logit": 4, "factored_probability": 5,
            "mlp_logit": 6, "mlp_probability": 7,
            "gate_logit": 8, "gate_probability": 9}
CODECS = {"fp32": 0, "fp16": 1, "int8": 2, "int4": 3}


def encode(family: str, peers: tuple[int, ...], arrays: list[np.ndarray], *,
           basis_rank: int = 0, feature_dim: int = 0, vocab: int = 0,
           basis_id: int = 0, codec: str = "fp32") -> bytes:
    if family not in FAMILIES or codec not in CODECS:
        raise ValueError("Unknown family or codec")
    if not peers or list(peers) != sorted(set(peers)) or min(peers) < 0 or max(peers) > 7:
        raise ValueError("Sorted unique peer IDs in 0..7 required")
    if any(not isinstance(x, int) or x < 0 or x >= 2**32
           for x in (basis_rank, feature_dim, vocab, basis_id)):
        raise ValueError("Metadata must be nonnegative uint32")
    payload = bytearray()
    for source in arrays:
        a = np.asarray(source, dtype=np.float32)
        if a.ndim not in (1, 2) or a.size == 0 or not np.isfinite(a).all():
            raise ValueError("Finite nonempty 1D/2D arrays required")
        rows, cols = (len(a), 0) if a.ndim == 1 else a.shape
        x = a.ravel()
        scale = 1.
        if codec in ("fp32", "fp16"):
            dtype = np.dtype("<f4" if codec == "fp32" else "<f2")
            if np.max(np.abs(x)) > np.finfo(dtype).max:
                raise ValueError("Floating point serialization overflow")
            raw = x.astype(dtype).tobytes()
        else:
            limit = 127 if codec == "int8" else 7
            scale = float(np.float32(float(np.max(np.abs(x))) / limit)) or 1.
            quant = np.rint(x / scale).clip(-limit, limit).astype(np.int8)
            if codec == "int8":
                raw = quant.tobytes()
            else:
                nibble = (quant.astype(np.int16) & 15).astype(np.uint8)
                if len(nibble) % 2:
                    nibble = np.r_[nibble, np.uint8(0)]
                raw = (nibble[::2] | (nibble[1::2] << 4)).tobytes()
        payload += ARRAY.pack(rows, cols, len(x), len(raw), scale)
        payload += raw
    return HEADER.pack(MAGIC, 1, FAMILIES[family], sum(1 << p for p in peers),
                       CODECS[codec], basis_rank, feature_dim, vocab, basis_id,
                       len(arrays), len(payload)) + payload


def decode(blob: bytes) -> dict:
    if len(blob) < HEADER.size:
        raise ValueError("Truncated header")
    magic, version, fid, mask, cid, rank, dim, vocab, basis_id, narr, size = HEADER.unpack_from(blob)
    if magic != MAGIC or version != 1 or mask == 0 or size != len(blob) - HEADER.size:
        raise ValueError("Invalid header")
    family = next((k for k, v in FAMILIES.items() if v == fid), None)
    codec = next((k for k, v in CODECS.items() if v == cid), None)
    if family is None or codec is None:
        raise ValueError("Unknown family/codec code")
    offset = HEADER.size
    arrays = []
    for _ in range(narr):
        if offset + ARRAY.size > len(blob):
            raise ValueError("Truncated array descriptor")
        rows, cols, n, nb, scale = ARRAY.unpack_from(blob, offset)
        offset += ARRAY.size
        expected = n * 4 if codec == "fp32" else n * 2 if codec == "fp16" else n if codec == "int8" else (n + 1) // 2
        if n == 0 or rows * (cols or 1) != n or nb != expected or offset + nb > len(blob):
            raise ValueError("Invalid array dimensions/payload size")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid quantization scale")
        raw = blob[offset:offset + nb]
        offset += nb
        if codec in ("fp32", "fp16"):
            x = np.frombuffer(raw, dtype="<f4" if codec == "fp32" else "<f2").astype(np.float32)
        elif codec == "int8":
            x = np.frombuffer(raw, dtype=np.int8).astype(np.float32) * scale
        else:
            packed = np.frombuffer(raw, dtype=np.uint8)
            v = np.empty(2 * len(packed), dtype=np.int16)
            v[::2] = packed & 15
            v[1::2] = packed >> 4
            x = np.where(v >= 8, v - 16, v)[:n].astype(np.float32) * scale
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite decoded parameters")
        arrays.append(x.reshape((rows, cols)) if cols else x)
    if offset != len(blob):
        raise ValueError("Trailing data")
    return {"family": family, "peers": tuple(p for p in range(8) if mask & (1 << p)),
            "arrays": arrays, "basis_rank": rank, "feature_dim": dim, "vocab": vocab,
            "basis_id": basis_id, "codec": codec, "actual_bytes": len(blob),
            "actual_bits": 8 * len(blob), "resident_parameter_bytes": sum(x.nbytes for x in arrays),
            "public_context_selector_scope": "basis_id identifies the registered basis AND feature/normalizer context"}


def export_residual(model, peers: tuple[int, ...], codec: str = "fp32", basis_id: int = 0) -> bytes:
    return export_model(model, peers, codec=codec, basis_id=basis_id)


def export_model(model, peers: tuple[int, ...], codec: str = "fp32", basis_id: int = 0,
                 *, vocab: int | None = None, retired_target: int | None = None) -> bytes:
    """Export target-specific coefficients; public basis arrays are not embedded.

    ``basis_id`` indexes a fixed, content-hashed public-context registry including
    the selected-peer feature transform/normalizer as well as the output basis.
    Any target-trained transform belongs in arrays and cannot use that registry.
    """
    from .models import (FrozenAnchor, TrainableCalibration, AnchoredResidual,
                         AnchoredMLP, AnchoredResponseGate)
    if retired_target is not None and retired_target in peers:
        raise ValueError("Retired target cannot occur among deployed peers")
    if isinstance(model, TrainableCalibration):
        model = model.frozen()
    to_numpy = lambda x: x.detach().cpu().numpy()
    if isinstance(model, FrozenAnchor):
        if vocab is None or vocab < 2:
            raise ValueError("Anchor export requires explicit output vocabulary size")
        if len(model.weights) != len(peers):
            raise ValueError("Peer metadata does not match the anchor")
        return encode("anchor_" + model.kind, peers, [to_numpy(model.weights)],
                      vocab=int(vocab), codec=codec)
    if len(model.anchor.weights) != len(peers):
        raise ValueError("Peer metadata does not match the anchor")
    kind = model.anchor.kind
    arrays = [to_numpy(model.anchor.weights)]
    if isinstance(model, AnchoredResidual):
        arrays += [to_numpy(model.bias)]
        if model.factor_rank is None:
            family = "linear_" + kind
            arrays += [to_numpy(model.linear)]
        else:
            family = "factored_" + kind
            arrays += [to_numpy(model.down), to_numpy(model.up)]
    elif isinstance(model, AnchoredMLP):
        family = "mlp_" + kind
        arrays += [to_numpy(model.hidden.weight), to_numpy(model.hidden.bias),
                   to_numpy(model.output.weight), to_numpy(model.output.bias)]
    elif isinstance(model, AnchoredResponseGate):
        if vocab is None or vocab < 2:
            raise ValueError("Gate export requires explicit output vocabulary size")
        arrays += [to_numpy(model.gate.weight), to_numpy(model.gate.bias),
                   np.asarray([model.strength], dtype=np.float32)]
        return encode("gate_" + kind, peers, arrays, feature_dim=model.feature_dim,
                      vocab=int(vocab), basis_id=basis_id, codec=codec)
    else:
        raise TypeError("Unsupported target-state model type")
    return encode(family, peers, arrays, basis_rank=model.basis.shape[1],
                  feature_dim=model.feature_dim, vocab=model.basis.shape[0],
                  basis_id=basis_id, codec=codec)


def predict_decoded(state: dict, peer_logits, features=None, basis=None):
    from .inference import predict_decoded as predict
    return predict(state, peer_logits, features, basis)
