"""Standalone adapters and literal byte encodings for the benchmarks experiment.

No fitting or state is imported from the original frozen experiment. The grids,
Helmert representation, RBF bandwidth and 22-byte header layout follow that
experiment. The new magic/family code distinguishes this corrected protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import hashlib
import math
import struct

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import nnls

GLOBAL_SEED = 20260722
LAMBDAS = tuple(float(x) for x in np.logspace(-8, 2, 17))
RANKS = (8, 16, 32, 64, 128, 256, 512, 1024)
WIDTHS = (1, 2, 3, 4, 6, 8, 12, 16, 32)
CODECS = tuple(f"q{x}" for x in WIDTHS) + ("native_fp16", "native_fp32", "native_fp64")
HEADER = struct.Struct("<4sBBBBBBHIBBI")
FAMILIES = {"convex": 1, "krr": 4, "nnls": 6}
PRECISIONS = {f"q{x}": x for x in WIDTHS} | {"native_fp16": 48, "native_fp32": 49, "native_fp64": 50}


def helmert(d: int) -> np.ndarray:
    h = np.zeros((d, d + 1))
    for i in range(d):
        h[i, : i + 1] = 1 / np.sqrt((i + 1) * (i + 2))
        h[i, i + 1] = -(i + 1) / np.sqrt((i + 1) * (i + 2))
    return h


def probabilities(coordinates: np.ndarray) -> np.ndarray:
    z = np.asarray(coordinates, dtype=float) @ helmert(coordinates.shape[-1])
    z -= z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def tv(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.abs(probabilities(target) - probabilities(prediction)).sum(axis=-1) / 2


def rbf(a: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    dist = np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T, 0)
    return np.exp(-dist / (2 * sigma * sigma))


@dataclass
class Context:
    pair: tuple[int, ...]
    mean: np.ndarray
    scale: np.ndarray
    sigma: float
    features: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    validation_projection: np.ndarray

    def project(self, peers: np.ndarray) -> np.ndarray:
        x = peers[:, self.pair, :].reshape(len(peers), -1)
        return rbf((x - self.mean) / self.scale, self.features, self.sigma) @ self.eigenvectors


def make_context(fit: np.ndarray, validation: np.ndarray, pair: tuple[int, ...]) -> Context:
    x = fit[:, pair, :].reshape(len(fit), -1)
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale < 1e-12, 1, scale)
    x = (x - mean) / scale
    rng = np.random.RandomState(GLOBAL_SEED)
    count = min(20000, len(x) * (len(x) - 1) // 2)
    left = rng.randint(0, len(x), size=max(count * 2, 1))
    right = rng.randint(0, len(x), size=max(count * 2, 1))
    keep = left != right
    distances = np.sqrt(((x[left[keep]] - x[right[keep]]) ** 2).sum(1))
    sigma = max(float(np.median(distances[:count])) if len(distances) else 1, 1e-8)
    eigenvalues, eigenvectors = eigh(rbf(x, x, sigma), driver="evd", check_finite=True)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    context = Context(pair, mean, scale, sigma, x, eigenvalues, eigenvectors, np.empty((0, 0)))
    context.validation_projection = context.project(validation)
    return context


def fit_weights(peers: np.ndarray, target: np.ndarray, pair: tuple[int, ...], family: str) -> np.ndarray:
    x = peers[:, pair, :].transpose(0, 2, 1).reshape(-1, len(pair))
    y = target.reshape(-1)
    if family == "nnls":
        return nnls(x, y)[0]
    if family != "convex":
        raise ValueError(f"Unknown weight family: {family}")
    # Enumerate the seven faces of a three-dimensional simplex. This solves
    # the small convex quadratic without optimizer-dependent stopping rules.
    candidates = []
    for size in range(1, len(pair) + 1):
        for support in combinations(range(len(pair)), size):
            a = x[:, support]
            system = np.block([[a.T @ a, np.ones((size, 1))], [np.ones((1, size)), np.zeros((1, 1))]])
            result = np.linalg.lstsq(system, np.r_[a.T @ y, 1], rcond=None)[0][:size]
            if result.min() >= -1e-9:
                w = np.zeros(len(pair))
                w[list(support)] = np.maximum(result, 0)
                w /= w.sum()
                candidates.append((float(np.square(x @ w - y).sum()), w))
    return min(candidates, key=lambda z: z[0])[1]


def _pack(values: np.ndarray, width: int) -> bytes:
    result = bytearray()
    accumulator = available = 0
    for value in values.reshape(-1):
        accumulator = (accumulator << width) | int(value)
        available += width
        while available >= 8:
            available -= 8
            result.append((accumulator >> available) & 255)
            accumulator &= (1 << available) - 1 if available else 0
    if available:
        result.append(accumulator << (8 - available))
    return bytes(result)


def _unpack(payload: bytes, count: int, width: int) -> np.ndarray:
    values = []
    accumulator = available = 0
    for byte in payload:
        accumulator = (accumulator << 8) | byte
        available += 8
        while available >= width and len(values) < count:
            available -= width
            values.append((accumulator >> available) & ((1 << width) - 1))
            accumulator &= (1 << available) - 1 if available else 0
    if len(values) != count:
        raise ValueError("Truncated payload")
    return np.asarray(values, dtype=np.uint64)


def _composition_rank(counts: np.ndarray, total: int) -> int:
    rank, remaining = 0, total
    for i, value in enumerate(counts[:-1]):
        later = len(counts) - i - 1
        value = int(value)
        rank += math.comb(remaining + later, later) - math.comb(remaining - value + later, later)
        remaining -= value
    return rank


def _composition_unrank(rank: int, total: int, parts: int) -> np.ndarray:
    counts, remaining = [], total
    for i in range(parts - 1):
        later = parts - i - 1
        lo, hi = 0, remaining
        while lo < hi:
            mid = (lo + hi + 1) // 2
            skipped = math.comb(remaining + later, later) - math.comb(remaining - mid + later, later)
            if skipped <= rank:
                lo = mid
            else:
                hi = mid - 1
        skipped = math.comb(remaining + later, later) - math.comb(remaining - lo + later, later)
        rank -= skipped
        counts.append(lo)
        remaining -= lo
    counts.append(remaining)
    if rank:
        raise ValueError("Invalid simplex codeword")
    return np.asarray(counts)


def state_bits(family: str, codec: str, rows: int, columns: int) -> int:
    if codec.startswith("native_fp"):
        payload = rows * columns * int(codec.removeprefix("native_fp"))
        scales = 0
    elif family == "convex":
        levels = (1 << int(codec[1:])) - 1
        payload = max(1, (math.comb(levels + rows - 1, rows - 1) - 1).bit_length())
        scales = 0
    else:
        payload, scales = rows * columns * int(codec[1:]), 32 * columns
    return HEADER.size * 8 + scales + 8 * ((payload + 7) // 8)


@dataclass(frozen=True)
class Encoded:
    blob: bytes
    family: str
    pair: tuple[int, ...]
    codec: str
    rank: int
    lambda_index: int
    meaningful_bits: int

    @property
    def bits(self) -> int:
        return 8 * len(self.blob)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.blob).hexdigest()


def encode(matrix: np.ndarray, family: str, pair: tuple[int, ...], codec: str, lambda_index: int = 0) -> Encoded:
    matrix = np.asarray(matrix, dtype=float).reshape(len(matrix), -1)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite state")
    rows, columns = matrix.shape
    scales = b""
    if family == "convex" and codec.startswith("q"):
        levels = (1 << int(codec[1:])) - 1
        w = np.maximum(matrix.reshape(-1), 0)
        w = w / w.sum()
        raw = w * levels
        counts = np.floor(raw).astype(np.int64)
        counts[np.argsort(-(raw - counts), kind="stable")[: levels - int(counts.sum())]] += 1
        size = math.comb(levels + rows - 1, rows - 1)
        payload_bits = max(1, (size - 1).bit_length())
        payload = _pack(np.asarray([_composition_rank(counts, levels)], dtype=object), payload_bits)
    elif codec.startswith("native_fp"):
        width = int(codec.removeprefix("native_fp"))
        dtype = np.dtype(f"<f{width // 8}")
        if np.max(np.abs(matrix), initial=0) > np.finfo(dtype).max:
            raise ValueError("Native precision overflow")
        payload = matrix.astype(dtype).tobytes()
        payload_bits = len(payload) * 8
    else:
        width = int(codec[1:])
        maximum = (1 << (width - 1)) - 1 if width > 1 else 1
        scale = np.max(np.abs(matrix), axis=0) / maximum
        scale = np.where(scale == 0, 1, scale)
        if np.max(scale) > np.finfo(np.float32).max:
            raise ValueError("Scale overflow")
        # The original encoder computes integers from float64 scales, then
        # transmits float32 scales. Reproduce that choice exactly.
        if width == 1:
            integers = (matrix >= 0).astype(np.uint64)
        else:
            integers = (np.clip(np.rint(matrix / scale), -maximum, maximum).astype(np.int64) + maximum).astype(np.uint64)
        scales = scale.astype("<f4").tobytes()
        payload = _pack(integers, width)
        payload_bits = matrix.size * width
    header = HEADER.pack(b"RBTL", 1, FAMILIES[family], PRECISIONS[codec], len(pair), sum(1 << p for p in pair), lambda_index, rows, rows, columns, 0, payload_bits)
    result = Encoded(header + scales + payload, family, pair, codec, rows, lambda_index, HEADER.size * 8 + len(scales) * 8 + payload_bits)
    if result.bits != state_bits(family, codec, rows, columns):
        raise AssertionError("State length mismatch")
    return result


def decode(blob: bytes) -> tuple[Encoded, np.ndarray]:
    if len(blob) < HEADER.size:
        raise ValueError("Truncated header")
    magic, version, family_code, precision, q, mask, li, rank, rows, columns, bias, payload_bits = HEADER.unpack(blob[: HEADER.size])
    if magic != b"RBTL" or version != 1 or bias != 0 or rank != rows:
        raise ValueError("Unsupported state header")
    family = {v: k for k, v in FAMILIES.items()}[family_code]
    codec = {v: k for k, v in PRECISIONS.items()}[precision]
    pair = tuple(i for i in range(8) if mask & (1 << i))
    if len(pair) != q or len(blob) * 8 != state_bits(family, codec, rows, columns):
        raise ValueError("State shape or byte count mismatch")
    offset = HEADER.size
    if family == "convex" and codec.startswith("q"):
        levels = (1 << int(codec[1:])) - 1
        code = int(_unpack(blob[offset:], 1, payload_bits)[0])
        matrix = (_composition_unrank(code, levels, q) / levels).reshape(rows, columns)
    elif codec.startswith("native_fp"):
        dtype = f"<f{int(codec.removeprefix('native_fp')) // 8}"
        matrix = np.frombuffer(blob[offset:], dtype=dtype).astype(float).reshape(rows, columns)
    else:
        scale = np.frombuffer(blob[offset: offset + columns * 4], dtype="<f4").astype(float)
        offset += columns * 4
        width = int(codec[1:])
        raw = _unpack(blob[offset:], rows * columns, width).reshape(rows, columns)
        matrix = (np.where(raw == 0, -1, 1) if width == 1 else raw.astype(np.int64) - ((1 << (width - 1)) - 1)) * scale
    meaningful = HEADER.size * 8 + (columns * 32 if codec.startswith("q") and family != "convex" else 0) + payload_bits
    padding = len(blob) * 8 - meaningful
    if padding < 0 or padding > 7 or (padding and blob[-1] & ((1 << padding) - 1)):
        raise ValueError("Invalid payload padding")
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite decoded state")
    if family == "convex":
        matrix = np.maximum(matrix, 0)
        matrix /= matrix.sum()
    # NNLS explicitly retains its scale; it is never simplex-normalized.
    return Encoded(blob, family, pair, codec, rank, li, meaningful), matrix


def predict(encoded: Encoded, matrix: np.ndarray, peers: np.ndarray, projection: np.ndarray | None = None) -> np.ndarray:
    if encoded.family == "krr":
        if projection is None:
            raise ValueError("KRR prediction requires its public peer projection")
        return projection[:, : encoded.rank] @ matrix
    return np.einsum("npd,p->nd", peers[:, encoded.pair, :], matrix.reshape(-1))


def paired_cluster_interval(difference: np.ndarray, clusters: np.ndarray, draws: int = 2000, seed: int = GLOBAL_SEED) -> tuple[float, float, float]:
    """Pairs predictions and resamples clusters, preserving item weighting."""
    if not len(difference) or len(difference) != len(clusters) or not np.isfinite(difference).all():
        raise ValueError("Invalid paired bootstrap inputs")
    if np.max(difference) == np.min(difference):
        return (float(difference[0]),) * 3
    unique, inverse = np.unique(clusters.astype(str), return_inverse=True)
    sums = np.bincount(inverse, weights=difference)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for start in range(0, draws, 128):
        ix = rng.integers(len(unique), size=(min(128, draws - start), len(unique)))
        estimates[start: start + len(ix)] = sums[ix].sum(1) / counts[ix].sum(1)
    return float(np.mean(difference)), float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))
