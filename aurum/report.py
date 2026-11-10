"""Build the final readable AURUM evidence package from completed results."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import pandas as pd
from .sprint import ROOT, dump, sha


def markdown(frame, columns=None, precision=6):
    frame = frame if columns is None else frame[columns]
    def value(x):
        if isinstance(x, (float, np.floating)): return f'{x:.{precision}f}'
        return str(x)
    return '| ' + ' | '.join(frame.columns) + ' |\n|' + '|'.join(['---'] * len(frame.columns)) + '|\n' + '\n'.join('| ' + ' | '.join(value(x) for x in row) + ' |' for row in frame.itertuples(index=False, name=None))


def build(catalog, confirmation, output):
    catalog = Path(catalog); confirmation = Path(confirmation); out = Path(output)
    if not (confirmation / 'complete.json').exists():
        raise ValueError('Final GOLD_REPORT requires completed independent confirmation')
    out.mkdir(parents=True, exist_ok=True)
    paired = pd.read_csv(confirmation / 'FIXED_CHOICE_PRICE.csv')
    summary = pd.read_csv(confirmation / 'CONFIRMATION_POLICY_SUMMARY.csv')
    gamma = paired[paired.comparator.eq('Gamma_best_fixed_pair') & paired.population.eq('mixture')]
    adaptation = paired[paired.comparator.str.startswith('cheap_pair_')].sort_values('difference_tv', ascending=False)
    strongest = adaptation.iloc[0]
    if (gamma.lower_simultaneous > 0).any():
        best = gamma.sort_values('difference_tv', ascending=False).iloc[0]
        headline = f"On fresh documents, target {int(best.target_index)} has a positive price of committing to one peer pair: budget-averaged TV penalty {best.difference_tv:.6f}, simultaneous 95% interval [{best.lower_simultaneous:.6f}, {best.upper_simultaneous:.6f}]."
    elif strongest.lower_simultaneous > 0:
        headline = f"Additional retained state improves distributional fidelity beyond fully reoptimized cheap calibration on {strongest.population} for target {int(strongest.target_index)}, {strongest.reference}: budget-averaged TV gain {strongest.difference_tv:.6f}, simultaneous 95% interval [{strongest.lower_simultaneous:.6f}, {strongest.upper_simultaneous:.6f}]. The complete rich family still admits a fixed peer winner."
    else:
        headline = f"The strongest fresh within-pair estimate is a budget-averaged TV gain of {strongest.difference_tv:.6f} on {strongest.population} for target {int(strongest.target_index)}, {strongest.reference}, with simultaneous 95% interval [{strongest.lower_simultaneous:.6f}, {strongest.upper_simultaneous:.6f}]. The study identifies a training-to-generalization gap and a stable fixed-peer regime; this interval limits the size of a supported adaptation claim."
    decisive = []
    for row in adaptation[adaptation.population.eq('mixture')].to_dict('records'):
        target = row['target_index']; base = summary[(summary.target_index == target) & (summary.policy == row['comparator'])]
        full = summary[(summary.target_index == target) & (summary.policy == row['reference'])]
        decisive.append({'target': target, 'pair': row['reference'].removeprefix('fixed_pair_'),
            'cheap weighted TV': base.tv.mean(), 'full weighted TV': full.tv.mean(),
            'gain': row['difference_tv'], 'simultaneous lower': row['lower_simultaneous'],
            'simultaneous upper': row['upper_simultaneous'], 'full bytes range': f'{int(full.actual_bytes.min())}–{int(full.actual_bytes.max())}',
            'argmax disagreement change': full.argmax_disagreement.mean() - base.argmax_disagreement.mean()})
    decisive = pd.DataFrame(decisive).sort_values(['target', 'pair'])
    decisive.to_csv(out / 'MAIN_RESULTS.csv', index=False)
    provenance = json.loads((confirmation / 'confirmation_provenance.json').read_text())
    files = {
        'ANCHOR_REPLAY.csv': ROOT / 'runs/aurum/ANCHOR_REPLAY.csv',
        'ANCHOR_LOSS_ABLATION.csv': ROOT / 'runs/aurum/ANCHOR_LOSS_ABLATION.csv',
        'SIX_CELL_SUMMARY.csv': ROOT / 'runs/aurum/SIX_CELL_SUMMARY.csv',
        'RICH_EXPANSION_SUMMARY.csv': ROOT / 'runs/aurum/RICH_EXPANSION_SUMMARY.csv',
        'ADAPTER_CAPACITY_CURVES.csv': catalog / 'ADAPTER_CAPACITY_CURVES.csv',
        'ALL_PEER_PAIRS.csv': catalog / 'ALL_PEER_PAIRS.csv',
        'ALL_CANDIDATES.csv': catalog / 'ALL_CANDIDATES.csv',
        'POLICY_DECISIONS.json': catalog / 'POLICY_DECISIONS.json',
        'CONFIRMATION_PER_CLUSTER.parquet': confirmation / 'CONFIRMATION_PER_CLUSTER.parquet',
        'FIXED_CHOICE_PRICE.csv': confirmation / 'FIXED_CHOICE_PRICE.csv',
        'CONFIRMATION_POLICY_SUMMARY.csv': confirmation / 'CONFIRMATION_POLICY_SUMMARY.csv',
        'MATCHED_FIDELITY_STATE.csv': confirmation / 'MATCHED_FIDELITY_STATE.csv',
        'MCQ_FIXED_CHOICE_PRICE_DEVELOPMENT.csv': ROOT / 'runs/aurum/mcq/FIXED_CHOICE_PRICE_DEVELOPMENT.csv',
        'DIAGNOSTIC_SUMMARY.csv': ROOT / 'runs/aurum/development/diagnostics/pair1-2_basis128_v1/DIAGNOSTIC_SUMMARY.csv',
        'ADAPTER_RESOURCE_MEASUREMENTS.csv': ROOT / 'runs/aurum/resources/final_gpu/ADAPTER_RESOURCE_MEASUREMENTS.csv',
        'MEASURED_COLLECTION_RESOURCES.csv': ROOT / 'runs/aurum/resources/new_development_collection/MEASURED_COLLECTION_RESOURCES.csv',
        'CONDITIONAL_PEER_FORWARD_SUMS.csv': ROOT / 'runs/aurum/resources/new_development_collection/CONDITIONAL_PEER_FORWARD_SUMS.csv',
    }
    for name in ('RICH_LEARNING_CURVE_SUMMARY.csv', 'RICH_SEED_SENSITIVITY.csv', 'RICH_EXPANSION_BY_SOURCE.csv'):
        source = ROOT / 'runs/aurum' / name
        if source.exists(): files[name] = source
    for name, source in files.items():
        if source.suffix in ('.csv', '.json'):
            text = source.read_text().replace(str(ROOT), '${REPO_ROOT}')
            (out / name).write_text(text)
        else: shutil.copy2(source, out / name)
    decisions = json.loads((catalog / 'POLICY_DECISIONS.json').read_text())
    resources = []
    for row in decisions:
        manifest = row.get('public_context_manifest'); public_disk = 0; registered_disk = 0
        if isinstance(manifest, str) and manifest:
            context = Path(manifest)
            if row['state_format'] == 'RBV2': public_disk = context.with_suffix('.npy').stat().st_size + context.stat().st_size
            else:
                meta = json.loads(context.read_text())
                # Training/validation projections are experiment cache, not deployment.
                public_disk = (context.parent / 'basis.npy').stat().st_size if (context.parent / 'basis.npy').exists() else 0
                public_disk += context.stat().st_size
            registered_disk = public_disk
            if row['state_format'] != 'RBV2':
                registered_disk = context.stat().st_size + sum((context.parent / name).stat().st_size for name in meta['files'])
        resources.append({k: row.get(k) for k in ('target_index', 'policy', 'cap_bytes', 'pair', 'candidate_id', 'actual_bytes', 'resident_parameter_bytes', 'public_numeric_bytes')} | {
            'public_basis_and_manifest_disk_bytes': public_disk,
            'registered_context_physical_disk_bytes_including_experiment_projection_cache': registered_disk,
            'public_normalizer_numeric_bytes_included_in_public_numeric_bytes': True,
            'peer_calls': 2, 'timing_scope': 'See separate measured adapter benchmark; encoded bytes are not serving VRAM or latency'})
    pd.DataFrame(resources).to_csv(out / 'PUBLIC_AND_TARGET_STATE.csv', index=False)
    diag = pd.read_csv(files['DIAGNOSTIC_SUMMARY.csv'])
    sprint = pd.read_csv(files['SIX_CELL_SUMMARY.csv'])
    report = f'''# AURUM: stronger adaptation, fixed-peer cost, and fresh confirmation

{headline}

## Decisive strong-control comparison

The table uses **{provenance['documents']} fresh document clusters**, half WikiText103 and half the pinned arXiv population. Target3 is Qwen3-8B; target4 is Qwen3-14B. Peers0/1/2 are Qwen3-0.6B/1.7B/4B. Each row compares the same fixed pair's fully retuned adapter menu with its independently retuned cheap menu. Values average equally over the eight declared budgets 128–2,097,152 bytes. The 42 contrasts across both targets and three predeclared populations (mixture and both sources) share one simultaneous 95% max-t family. The argmax column is full minus cheap, so positive means worse decision agreement.

{markdown(decisive)}

The distinct peer-selection endpoint is Γ, the minimum penalty over **every** admissible fixed pair. Each fixed pair freely selects its adapter, objective, rank, regularization and precision at every budget from the very same catalog used by joint selection.

{markdown(gamma, ['target_index', 'difference_tv', 'lower_simultaneous', 'upper_simultaneous'])}

The full tables retain every fixed pair, native-full, unbounded-menu, mean-budget and minimax-regret policy, and joint-cheap. At one known budget, joint cannot strictly beat the fixed pair that it selected; Γ measures the cost of committing before the budget is specified. Equal log-budget weighting is a research estimand, not an observed production workload.

## Predeclared source-specific adaptation contrasts

Both source populations were selected for confirmation before collecting their responses, motivated by development heterogeneity. Their intervals belong to the same42-contrast family above.

{markdown(adaptation[adaptation.population.ne('mixture')], ['target_index', 'population', 'comparator', 'difference_tv', 'lower_simultaneous', 'upper_simultaneous'])}

## What the experiments explain

The first six cells isolated legacy NNLS versus an inner-selected cheap anchor, and projected-logit MSE versus full-support KL, followed by TV refinement. Original decoded convex/NNLS states replayed exactly on the original GPU evaluation setup. A better starting prediction and probability objectives removed much of the old residual degradation. Against the subsequently strengthened cheap calibration, first-sprint improvements ranged from negative values to about 0.0013 TV. These original-cache numbers remain development evidence.

The output basis can represent substantially more correction than transfers to held-out documents. On 32 fixed inner-fit examples, rank 128 linear maps reached TV 0.0260 for target4 and 0.0384 for target3, from anchors 0.2754 and 0.2618. Independent per-query coefficients reached 0.0183 and 0.0324. These are deliberately nondeployable or memorization diagnostics, not held-out gains or certified lower bounds. Together with the much smaller outer-validation gains they motivated compact mappings, regularization and data scaling rather than indiscriminate basis enlargement.

The expanded family gave every pair direct bounded KL/TV calibration, each single-peer temperature control, linear ranks 16/32/64/128, factor ranks 1/2/4/8/16, response-conditioned gates and MLP widths 16/32/64. Four literal codecs were scored after decoding. Thirty initial architecture/optimizer configurations per cell were screened on inner stopping, with continued optimization and TV refinement for each architecture family. Original RBV2 states remain eligible at their original byte sizes.

New development collection added 2048 previously unused documents. Learning curves compare 512,1024,1364 and3412 fitting documents while keeping the original peer-only basis, normalizers, inner stopping and outer validation fixed. The main catalog uses the declared seed 20261110; two extra seed/split runs are sensitivity analyses and are not selected as the best seed or counted as independent documents. All completed candidates, including worsening ones, remain in the ledgers.

The motivated heterogeneous extension evaluated all 180 target/task/pair cells across six models and three MCQ tasks. Seventeen task-target grids had Γ=0 on the128–8192-byte grid. The lone development crossing, HellaSwag/Qwen3.5-9B, had Γ≈0.000380 with paired cluster SD≈0.07084 on 512 validation clusters. This is too small and noisy to establish a useful resource consequence from that cached grid. Its complete competitors and earlier KRR states remain visible; this extension is not independent confirmation of the rich experiment.

## Independent check and resource scope

The method lock hashes the complete chosen state family, policy decisions, comparison family, inference code, population and budget weights before confirmation documents are selected or model responses collected. Fresh membership excludes all previous pilot/fit/validation/test cohorts and the2048 new development documents. Per-cluster results join to policies by target, pair, candidate ID and state format, so repeated budgets never become extra independent samples. Confirmation cohort hash: `{provenance['cohort_sha256']}`. Method-lock hash: `{provenance['method_lock_sha256']}`.

`PUBLIC_AND_TARGET_STATE.csv` separates actual encoded target bytes, decoded resident parameters, shared basis/normalizers and public files. A rank 128 full-vocabulary FP32 basis alone has 77,791,232 numeric bytes. The128-byte through2MiB target-state grid does not imply a corresponding total serving-memory budget. `MATCHED_FIDELITY_STATE.csv` retains all development-chosen thresholds and infeasible cases; threshold claims require the reported conservative simultaneous risk upper bound. No production storage limit or unmeasured latency saving is inferred from a compression ratio.

Adapter timings were measured on an NVIDIA B200 for 50 repetitions at batch sizes 1/16/64, including selected-peer feature projection, normalization, correction and full-vocabulary log-softmax. They exclude peer-model forward calls. The separate new-development collection ledger measures actual BF16 model forward times and GPU peaks; sums of separately observed peer times are conditional cost estimates, not an end-to-end serving benchmark. These measurements are in `ADAPTER_RESOURCE_MEASUREMENTS.csv`, `MEASURED_COLLECTION_RESOURCES.csv` and `CONDITIONAL_PEER_FORWARD_SUMS.csv`.

## Reproducibility and scope

`CLAIM_TO_EVIDENCE.csv` maps each claim to its table and reproduction command. `ALL_CANDIDATES.csv`, the original first sprint, complete heterogeneous search and training histories retain positive, zero and negative outcomes. The source, binary-state inventory, data provenance and exact locks are available in the companion reproducibility artifact; large logits remain in separately hashed local banks.

This study supports only the displayed fresh effects within this fitted candidate family and declared document population. Within-pair adaptation, cheap calibration improvements and the price of fixed peers are different findings. A stable fixed winner restricts the empirical coupling claim; an optimizer or anchor improvement does not establish a new selection algorithm, an all-encoder lower bound or a Gaussian spectral mechanism.
'''
    (out / 'GOLD_REPORT.md').write_text(report)
    claims = []
    for row in paired.to_dict('records'):
        claims.append({'claim': row['comparator'] + ' minus ' + row['reference'], 'scope': f"rich target{row['target_index']}; {row['population']}; equal log-budget average; fresh documents",
            'effect_tv': row['difference_tv'], 'simultaneous_lower': row['lower_simultaneous'], 'simultaneous_upper': row['upper_simultaneous'],
            'source_artifact': 'FIXED_CHOICE_PRICE.csv + CONFIRMATION_PER_CLUSTER.parquet + POLICY_DECISIONS.json',
            'reproduce_command': 'python -m aurum.confirmation analyze --method-lock runs/aurum/confirmation_lock.json --output runs/aurum/confirmation/results',
            'caveat': 'Fitted family only; asymptotic document-cluster intervals; no physical latency/state-saving claim'})
    claims += [
        {'claim': 'Rank128 basis and linear map can memorize32 inner-fit documents', 'scope': 'Fit-only diagnostic, pair1,2, targets3/4', 'source_artifact': 'DIAGNOSTIC_SUMMARY.csv', 'reproduce_command': 'python -m aurum.diagnose --help', 'caveat': 'Neither held-out evidence nor a deployable per-query oracle'},
        {'claim': 'Heterogeneous MCQ grid has one tiny development fixed-choice crossing', 'scope': '180 cells; six models; three tasks; existing development cache', 'effect_tv': .000380, 'source_artifact': 'MCQ_FIXED_CHOICE_PRICE_DEVELOPMENT.csv', 'reproduce_command': 'python -m aurum.mcq_analysis --output runs/aurum/mcq --replay', 'caveat': 'Exploratory selection estimate; not independently confirmed'},
        {'claim': 'Measured adapter state, public memory and latency', 'scope': 'NVIDIA B200; batches1/16/64;50 repetitions', 'source_artifact': 'ADAPTER_RESOURCE_MEASUREMENTS.csv + PUBLIC_AND_TARGET_STATE.csv', 'reproduce_command': 'python -m aurum.resource_benchmark --catalog runs/aurum/final_catalog/POLICY_DECISIONS.json --output runs/aurum/resources/reproduction --device cuda:0', 'caveat': 'Peer-model forward calls excluded; target bytes do not measure total serving resources'},
    ]
    pd.DataFrame(claims).to_csv(out / 'CLAIM_TO_EVIDENCE.csv', index=False)
    history = [
        ('anchor_replay', '12 legacy cheap states, all ten development banks', 'Exact GPU replay; source/cache alignment', 'Run controlled anchor/loss factorial'),
        ('anchor_loss_factorial', 'Both targets × all3pairs,32/128 FP32', 'Small mixed benefits beyond strong calibration; stablepair12', 'Diagnose representation versus generalization'),
        ('tiny_fit_oracle', '32 inner-fit examples per target,pair12', 'Basis/map can memorize; no held-out claim', 'Regularize mappings and strengthen cheap optimization'),
        ('expanded_rich', '6cells,1464decodedcandidate rows', 'Compact nonlinear families and stronger scalar controls evaluated equally', 'Collect newdevelopment forlearningcurve'),
        ('heterogeneous_mcq', '180completecells; allremainingpeerpairs', 'One tiny/noisyΓcrossing;17fixed-winnergrids', 'Retain complete scope; prioritize richgeneralization confirmation'),
        ('additional_development', '2048newdocuments×5models', 'Independent ofpreviouscohorts; usedfortraining only', 'Learningcurve512/1024/1364/3412'),
        ('seed_sensitivity', 'Three declared fit/split seeds atlargesttrainingN', 'Sensitivity results separate; mainseed fixed', 'Freeze complete maincatalog'),
        ('confirmation', str(provenance['documents'])+'freshclusters', 'Alllockedstates evaluated together; allsigned contrasts retained', 'Report mechanism,effects,scope'),
    ]
    pd.DataFrame(history, columns=['branch','scope','finding_or_purpose','next_action']).to_csv(out / 'SEARCH_HISTORY.csv', index=False)
    (out / 'README.md').write_text('Read GOLD_REPORT.md first. MAIN_RESULTS.csv is the concise fresh strong-control table. All reported effects are TV differences; positive fixed-minus-joint or cheap-minus-full favors the reference. Individual text/CSV files are provided alongside the ZIP.\n')
    manifest = [{'file': str(p.relative_to(out)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'MANIFEST.json']
    dump(out / 'MANIFEST.json', manifest)
    archive = out.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(out))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(json.dumps({'directory': str(out), 'archive': str(archive), 'archive_bytes': archive.stat().st_size, 'archive_sha256': sha(archive)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--catalog', required=True); p.add_argument('--confirmation', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); build(a.catalog, a.confirmation, a.output)
