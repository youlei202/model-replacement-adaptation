#!/usr/bin/env python3
"""Posthoc CPU accounting of already-frozen MCQ deployments; no reselection."""
import argparse
import gc
import hashlib
import json
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcq_deployment import DeployedAdapter, core, sha


def deep_size(obj, seen=None):
    """Identity-deduplicated Python graph size, including ndarray backing stores.

    Owning NumPy arrays include their buffer in getsizeof; views additionally
    retain their base. Modules, classes, code, and unreferenced allocator arenas
    are not deployment-owned objects and are deliberately not traversed.
    """
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(deep_size(k, seen) + deep_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (tuple, list, set, frozenset)):
        size += sum(deep_size(v, seen) for v in obj)
    elif isinstance(obj, np.ndarray):
        if obj.base is not None:
            size += deep_size(obj.base, seen)
        if obj.dtype.hasobject:
            raise ValueError('Object arrays require a different accounting rule')
    else:
        if hasattr(obj, '__dict__'):
            size += deep_size(vars(obj), seen)
        for cls in type(obj).__mro__:
            slots = cls.__dict__.get('__slots__', ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots:
                if slot not in ('__dict__', '__weakref__') and hasattr(obj, slot):
                    size += deep_size(getattr(obj, slot), seen)
    return size


def trace_factory(factory, repetitions):
    # Warm import caches and Python key-sharing dictionaries before measurement.
    for _ in range(3):
        warm = factory()
        del warm
    records = []
    for _ in range(repetitions):
        gc.collect()
        tracemalloc.start()
        before, _ = tracemalloc.get_traced_memory()
        value = factory()
        after, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        records.append({'retained_net_bytes': after - before,
                        'transient_peak_bytes': peak - before})
        del value
    return {'repetitions': repetitions,
            'retained_net_median_bytes': statistics.median(r['retained_net_bytes'] for r in records),
            'transient_peak_median_bytes': statistics.median(r['transient_peak_bytes'] for r in records),
            'raw_repetitions': records}


def inputs(spec):
    paths = {Path(spec)}
    for row in json.loads(Path(spec).read_text())['systems']:
        paths.update(p for p in Path(row['bundle']).iterdir() if p.is_file())
    for row in json.loads(Path(spec).read_text())['systems']:
        manifest = json.loads((Path(row['bundle']) / 'manifest.json').read_text())
        for name in manifest['code_sha256']:
            source = Path(core.__file__) if name == Path(core.__file__).name else Path(__file__).parent / name
            paths.add(source)
    return {str(p.resolve()): sha(p) for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, default=Path(__file__).parent / 'runs/v2/resource_benchmark/confirmation_catalog/final_resource_spec.json')
    parser.add_argument('--repetitions', type=int, default=20)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.repetitions < 3:
        raise ValueError('At least three accounting repetitions required')
    output = args.output or args.spec.with_name('resident_state_supplement.json')
    if output.exists():
        raise FileExistsError(f'Preserve the existing posthoc audit: {output}')
    before = inputs(args.spec)
    measurements = []
    for row in json.loads(args.spec.read_text())['systems']:
        bundle = Path(row['bundle'])
        blob = (bundle / 'state.bin').read_bytes()
        decoded_trace = trace_factory(lambda: core.decode(blob), args.repetitions)
        constructor_trace = trace_factory(lambda: DeployedAdapter(bundle), args.repetitions)
        adapter = DeployedAdapter(bundle)
        manifest = adapter.manifest
        # One disjoint partition of the whole adapter graph; shared references
        # are assigned to the first listed component, never charged twice.
        seen = set()
        decoded = deep_size(adapter.encoded, seen) + deep_size(adapter.matrix, seen)
        shared = deep_size(adapter.context, seen) if adapter.context is not None else 0
        routing = deep_size(adapter.manifest, seen)
        wrapper = deep_size(adapter, seen)
        total = decoded + shared + routing + wrapper
        assert total == deep_size(adapter), 'Object graph partition does not sum'
        shared_numeric = sum(v.nbytes for v in vars(adapter.context).values()) if adapter.context else 0
        assert adapter.matrix.nbytes == manifest['decoded_coefficient_bytes']
        assert shared_numeric == manifest['public_arrays_resident_bytes']
        assert len(blob) == manifest['state_binary_bytes']
        arrays = ({name: {'shape': list(value.shape), 'dtype': str(value.dtype),
                         'numeric_payload_bytes': value.nbytes}
                   for name, value in vars(adapter.context).items()} if adapter.context else {})
        measurements.append({
            'system_id': row['system_id'], 'target_index': manifest['target_index'],
            'adapter_family': manifest['adapter_family'], 'peer_indices': list(adapter.pair),
            'state_sha256': sha(bundle / 'state.bin'),
            'manifest_sha256': sha(bundle / 'manifest.json'),
            'shared_context_sha256': manifest['public_context_sha256'],
            'serialized_literal_state_bytes': len(blob),
            'serialized_state_python_bytes_object_bytes': sys.getsizeof(adapter.encoded.blob),
            'serialized_routing_manifest_bytes': (bundle / 'manifest.json').stat().st_size,
            'serialized_shared_context_bytes': adapter.accounting['shared_deployment_disk_bytes'],
            'decoded_coefficient_numeric_payload_bytes': adapter.matrix.nbytes,
            'retained_encoded_plus_coefficient_numeric_payload_bytes': len(blob) + adapter.matrix.nbytes,
            'decoded_state_python_graph_bytes_including_encoded_blob': decoded,
            'shared_context_numeric_payload_bytes': shared_numeric,
            'shared_context_python_graph_exclusive_bytes': shared,
            'routing_manifest_python_graph_exclusive_bytes': routing,
            'adapter_wrapper_path_accounting_python_graph_exclusive_bytes': wrapper,
            'complete_adapter_python_graph_bytes_excluding_models_and_shared_runtime': total,
            'target_state_and_routing_python_graph_bytes_excluding_shared_context': total - shared,
            'shared_arrays': arrays,
            'decode_only_trace': decoded_trace,
            'complete_adapter_construction_trace': constructor_trace,
            'models_loaded': False, 'GPU_used': False,
        })
    after = inputs(args.spec)
    assert before == after, 'Frozen audit inputs changed'
    result = {
        'schema': 'mcq_resident_state_posthoc_cpu_v1',
        'created_unix': time.time(),
        'posthoc_accounting_only_no_reselection': True,
        'frozen_inputs_verified_unchanged': True,
        'execution_device': 'CPU only; no torch or model instantiation',
        'environment': {'python': sys.version, 'numpy': np.__version__,
                        'platform': platform.platform(), 'pointer_bytes': np.dtype(np.intp).itemsize},
        'input_sha256': before, 'audit_code_sha256': sha(__file__),
        'measurements': measurements,
        'legacy_field_correction': 'resident_target_coefficients_bytes + resident_target_encoded_bytes is numerical/serialized payload only, not full Python resident memory. Shared deployment resident bytes likewise counts array payload only.',
        'graph_partition_order': ['decoded state including retained encoded blob', 'shared context', 'routing manifest', 'adapter wrapper/path/accounting'],
        'measurement_scope': {
            'deep_graph': 'sys.getsizeof recursively over object attributes, slots, containers and ndarray bases; identity-deduplicated. Includes referenced interned integers/strings and shared key strings conservatively. Does not recurse into class/module/code objects, NumPy dtype internals, malloc arenas or model weights.',
            'decode_trace': 'Input serialized bytes allocated before tracing and excluded from incremental decode trace. They remain retained by Encoded and are separately charged. Return tuple is included in this isolated trace although DeployedAdapter stores its two elements separately.',
            'constructor_trace': 'Complete DeployedAdapter construction including JSON/NPZ loading and integrity checks. Peak includes transient file/hash buffers, not just inference state. Warm-up imports/caches excluded. Filesystem cache already warm.',
            'trace_limit': 'tracemalloc observes allocations registered with Python tracing; native allocations may be absent. Measurements depend on Python/NumPy versions and allocator state. Neither graph nor traced bytes is incremental RSS or a memory-feasibility certificate.',
            'shared_vs_conditional': 'Shared public context is separately accounted and charged once per deployed service; distinct services may duplicate it. No unmeasured cross-service deduplication is assumed. Standalone service additionally includes callable model weights and runtime; their actual process RSS/VRAM is in live benchmark traces.',
            'not_measured_here': ['model weights', 'tokenizers', 'CUDA allocations', 'inference intermediates', 'process RSS', 'service concurrency/capacity'],
        },
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps([{k: row[k] for k in ('system_id', 'serialized_literal_state_bytes',
          'decoded_coefficient_numeric_payload_bytes', 'decoded_state_python_graph_bytes_including_encoded_blob',
          'shared_context_python_graph_exclusive_bytes',
          'complete_adapter_python_graph_bytes_excluding_models_and_shared_runtime')}
          for row in measurements], indent=2))


if __name__ == '__main__':
    main()
