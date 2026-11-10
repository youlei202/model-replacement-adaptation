from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .adapters import AdapterContext, AdapterState


MAGIC = b"KISQ"
VERSION = 1
HEADER = struct.Struct("<4sBBBBBBHIBBI")
ADAPTER_CODES = {"convex": 1, "linear": 2, "rff": 3, "krr": 4, "mlp": 5}
ADAPTER_NAMES = {value: key for key, value in ADAPTER_CODES.items()}
PRECISIONS = {
    "q1": 1,
    "q2": 2,
    "q3": 3,
    "q4": 4,
    "q6": 6,
    "q8": 8,
    "q12": 12,
    "q16": 16,
    "q32": 32,
    "native_fp16": 48,
    "native_fp32": 49,
    "native_fp64": 50,
}
PRECISION_NAMES = {value: key for key, value in PRECISIONS.items()}


@dataclass(frozen=True)
class EncodedState:
    bitstream: bytes
    complete_bits: int
    byte_padding_bits: int
    bitstream_digest: str
    codec: str
    rank: int


def _peer_mask(indices: Tuple[int, ...]) -> int:
    mask = 0
    for index in indices:
        mask |= 1 << int(index)
    return mask


def _peer_indices(mask: int) -> Tuple[int, ...]:
    return tuple(index for index in range(8) if mask & (1 << index))


def _matrix(state: AdapterState, context: AdapterContext, rank: int) -> Tuple[np.ndarray, bool]:
    if state.adapter == "convex":
        return np.asarray(state.weights, dtype=np.float64).reshape(-1, 1), False
    if state.adapter in ("linear", "rff"):
        coefficient = np.asarray(state.coefficient, dtype=np.float64)
        if rank > 0:
            coefficient = coefficient[: min(rank, len(coefficient))]
        bias = np.asarray(state.bias, dtype=np.float64).reshape(1, -1)
        return np.vstack([coefficient, bias]), True
    if state.adapter == "krr":
        if context.eigvecs is None:
            raise ValueError("KRR codec requires public eigenvectors")
        spectral = context.eigvecs.T @ np.asarray(state.alpha, dtype=np.float64)
        selected_rank = min(rank if rank > 0 else len(spectral), len(spectral))
        return spectral[:selected_rank], False
    if state.adapter == "mlp":
        parameters = np.concatenate(
            [
                np.asarray(state.hidden_weight, dtype=np.float64).reshape(-1),
                np.asarray(state.hidden_bias, dtype=np.float64).reshape(-1),
                np.asarray(state.output_weight, dtype=np.float64).reshape(-1),
                np.asarray(state.output_bias, dtype=np.float64).reshape(-1),
            ]
        )
        return parameters.reshape(-1, 1), False
    raise ValueError("unsupported adapter %s" % state.adapter)


def _pack_bits(values: np.ndarray, width: int) -> Tuple[bytes, int]:
    output = bytearray()
    accumulator = 0
    count = 0
    for raw in values.reshape(-1):
        accumulator = (accumulator << width) | int(raw)
        count += width
        while count >= 8:
            count -= 8
            output.append((accumulator >> count) & 0xFF)
            accumulator &= (1 << count) - 1 if count else 0
    if count:
        output.append((accumulator << (8 - count)) & 0xFF)
    return bytes(output), len(values.reshape(-1)) * width


def _unpack_bits(payload: bytes, count: int, width: int) -> np.ndarray:
    values: List[int] = []
    accumulator = 0
    available = 0
    for byte in payload:
        accumulator = (accumulator << 8) | byte
        available += 8
        while available >= width and len(values) < count:
            available -= width
            values.append((accumulator >> available) & ((1 << width) - 1))
            accumulator &= (1 << available) - 1 if available else 0
    if len(values) != count:
        raise ValueError("truncated quantized payload")
    return np.asarray(values, dtype=np.uint64)


def _simplex_counts(weights: np.ndarray, levels: int) -> np.ndarray:
    scaled = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    scaled = scaled / scaled.sum() if scaled.sum() else np.ones(len(scaled)) / len(scaled)
    raw = scaled * levels
    counts = np.floor(raw).astype(np.int64)
    remainder = levels - int(counts.sum())
    order = np.argsort(-(raw - counts))
    counts[order[:remainder]] += 1
    return counts


def _composition_rank(counts: np.ndarray, total: int) -> int:
    rank = 0
    remaining = total
    parts = len(counts)
    for index in range(parts - 1):
        later = parts - index - 1
        for value in range(int(counts[index])):
            rank += math.comb(remaining - value + later - 1, later - 1)
        remaining -= int(counts[index])
    return rank


def _composition_unrank(rank: int, total: int, parts: int) -> np.ndarray:
    counts = []
    remaining = total
    current_rank = int(rank)
    for index in range(parts - 1):
        later = parts - index - 1
        value = 0
        while value <= remaining:
            block = math.comb(remaining - value + later - 1, later - 1)
            if current_rank < block:
                break
            current_rank -= block
            value += 1
        counts.append(value)
        remaining -= value
    counts.append(remaining)
    return np.asarray(counts, dtype=np.int64)


def encode_state(state: AdapterState, context: AdapterContext, codec: str, rank: int = 0) -> EncodedState:
    if codec not in PRECISIONS:
        raise ValueError("unknown codec %s" % codec)
    matrix, has_bias = _matrix(state, context, rank)
    if not np.isfinite(matrix).all():
        raise ValueError("state matrix contains non-finite values")
    rows, columns = matrix.shape
    selected_rank = int(state.rank) if state.adapter == "mlp" else rows - int(has_bias)
    precision = PRECISIONS[codec]
    scales = b""
    if state.adapter == "convex" and codec.startswith("q"):
        width = int(codec[1:])
        levels = (1 << width) - 1
        counts = _simplex_counts(matrix.reshape(-1), levels)
        codebook_size = math.comb(levels + len(counts) - 1, len(counts) - 1)
        payload_bits = max(1, int(math.ceil(math.log(codebook_size, 2))))
        codeword = _composition_rank(counts, levels)
        payload, _ = _pack_bits(np.asarray([codeword], dtype=object), payload_bits)
    elif codec.startswith("native_fp"):
        dtype = {"native_fp16": "<f2", "native_fp32": "<f4", "native_fp64": "<f8"}[codec]
        limit = np.finfo(np.dtype(dtype)).max
        if np.max(np.abs(matrix), initial=0.0) > limit:
            raise ValueError("state matrix cannot be represented by %s" % codec)
        payload = np.asarray(matrix, dtype=dtype).tobytes(order="C")
        payload_bits = len(payload) * 8
    else:
        width = int(codec[1:])
        quantized = np.zeros(matrix.shape, dtype=np.uint64)
        scale_values = np.zeros(columns, dtype="<f4")
        if width == 1:
            for column in range(columns):
                scale = float(np.max(np.abs(matrix[:, column]))) or 1.0
                if not np.isfinite(scale) or scale > np.finfo(np.float32).max:
                    raise ValueError("quantization scale cannot be represented by float32")
                scale_values[column] = scale
                quantized[:, column] = (matrix[:, column] >= 0.0).astype(np.uint64)
        else:
            maximum = (1 << (width - 1)) - 1
            for column in range(columns):
                scale = float(np.max(np.abs(matrix[:, column]))) / maximum or 1.0
                if not np.isfinite(scale) or scale > np.finfo(np.float32).max:
                    raise ValueError("quantization scale cannot be represented by float32")
                scale_values[column] = scale
                signed = np.clip(np.rint(matrix[:, column] / scale), -maximum, maximum).astype(np.int64)
                quantized[:, column] = (signed + maximum).astype(np.uint64)
        scales = scale_values.tobytes(order="C")
        payload, payload_bits = _pack_bits(quantized, width)
    header = HEADER.pack(
        MAGIC,
        VERSION,
        ADAPTER_CODES[state.adapter],
        precision,
        len(state.peer_indices),
        _peer_mask(state.peer_indices),
        int(state.lambda_index),
        int(selected_rank),
        int(rows),
        int(columns),
        int(has_bias),
        int(payload_bits),
    )
    bitstream = header + scales + payload
    complete_bits = len(header) * 8 + len(scales) * 8 + payload_bits
    padding = len(bitstream) * 8 - complete_bits
    if padding and bitstream[-1] & ((1 << padding) - 1):
        raise RuntimeError("nonzero bitstream padding")
    return EncodedState(bitstream, complete_bits, padding, hashlib.sha256(bitstream).hexdigest(), codec, selected_rank)


def decode_state(encoded: EncodedState, context: AdapterContext) -> AdapterState:
    fields = HEADER.unpack(encoded.bitstream[: HEADER.size])
    magic, version, adapter_code, precision, q, mask, lambda_index, rank, rows, columns, has_bias, payload_bits = fields
    if magic != MAGIC or version != VERSION:
        raise ValueError("invalid state bitstream")
    indices = _peer_indices(mask)
    if len(indices) != q:
        raise ValueError("peer subset metadata is inconsistent")
    codec = PRECISION_NAMES[precision]
    offset = HEADER.size
    if adapter_code == ADAPTER_CODES["convex"] and codec.startswith("q"):
        width = int(codec[1:])
        levels = (1 << width) - 1
        payload = encoded.bitstream[offset : offset + (payload_bits + 7) // 8]
        codeword = int(_unpack_bits(payload, 1, payload_bits)[0])
        counts = _composition_unrank(codeword, levels, q)
        matrix = (counts / float(levels)).reshape(q, 1)
    elif codec.startswith("native_fp"):
        dtype = {"native_fp16": "<f2", "native_fp32": "<f4", "native_fp64": "<f8"}[codec]
        size = np.dtype(dtype).itemsize * rows * columns
        matrix = (
            np.frombuffer(encoded.bitstream[offset : offset + size], dtype=dtype)
            .astype(np.float64)
            .reshape(rows, columns)
        )
    else:
        width = int(codec[1:])
        scale_bytes = columns * 4
        scales = np.frombuffer(encoded.bitstream[offset : offset + scale_bytes], dtype="<f4").astype(np.float64)
        offset += scale_bytes
        payload = encoded.bitstream[offset : offset + (payload_bits + 7) // 8]
        raw = _unpack_bits(payload, rows * columns, width).reshape(rows, columns)
        if width == 1:
            matrix = np.where(raw == 0, -1.0, 1.0) * scales.reshape(1, -1)
        else:
            maximum = (1 << (width - 1)) - 1
            matrix = (raw.astype(np.int64) - maximum) * scales.reshape(1, -1)
    if not np.isfinite(matrix).all():
        raise ValueError("decoded state matrix contains non-finite values")
    adapter = ADAPTER_NAMES[adapter_code]
    state = AdapterState(adapter, indices, int(lambda_index), int(rank))
    if adapter == "convex":
        state.weights = matrix.reshape(-1)
        state.weights = np.clip(state.weights, 0.0, None)
        state.weights = state.weights / state.weights.sum() if state.weights.sum() else np.ones(q) / q
    elif adapter in ("linear", "rff"):
        state.coefficient = matrix[:-1] if has_bias else matrix
        state.bias = matrix[-1] if has_bias else np.zeros(columns)
        state.rank = len(state.coefficient)
    elif adapter == "krr":
        if context.eigvecs is None:
            raise ValueError("KRR decoder requires public eigenvectors")
        state.alpha = context.eigvecs[:, :rank] @ matrix[:rank]
        state.rank = rank
    else:
        input_dimension = int(context.fit_features.shape[1])
        output_dimension = int(context.output_dimension)
        hidden_width = int(rank)
        flat = matrix.reshape(-1)
        expected = input_dimension * hidden_width + hidden_width + hidden_width * output_dimension + output_dimension
        if len(flat) != expected:
            raise ValueError("MLP payload has %d parameters; expected %d" % (len(flat), expected))
        offset = 0
        state.hidden_weight = flat[offset : offset + input_dimension * hidden_width].reshape(
            input_dimension, hidden_width
        )
        offset += input_dimension * hidden_width
        state.hidden_bias = flat[offset : offset + hidden_width]
        offset += hidden_width
        state.output_weight = flat[offset : offset + hidden_width * output_dimension].reshape(
            hidden_width, output_dimension
        )
        offset += hidden_width * output_dimension
        state.output_bias = flat[offset : offset + output_dimension]
        state.metadata = {"activation": "tanh"}
    return state
