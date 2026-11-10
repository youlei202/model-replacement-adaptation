"""Classify the evidence actually measured; never force a favorable verdict."""
from __future__ import annotations
from math import isfinite
from typing import Mapping


INTEGRITY_FIELDS = (
    'complete_candidate_family', 'target_excluded', 'strong_baselines_complete',
    'actual_state_replay_passed', 'freshness_audit_passed', 'freeze_verified',
    'multiplicity_reported', 'all_outcomes_reported',
)


def _number(report: Mapping, field: str) -> float | None:
    value = report.get(field)
    if isinstance(value, bool) or value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if isfinite(f) else None


def classify(report: Mapping) -> dict:
    missing = [key for key in INTEGRITY_FIELDS if report.get(key) is not True]
    if missing:
        return {'level': 'INCOMPLETE_EVIDENCE', 'missing': missing,
                'message': 'Execution may be useful, but required evidence checks are incomplete.'}
    gain, gain_low = _number(report, 'gain_vs_strongest_fixed'), _number(report, 'simultaneous_gain_low')
    decision = (gain is not None and gain >= .005 and gain_low is not None and gain_low > 0
                and report.get('all_claimed_fixed_comparators_passed') is True)
    saving, saving_low = _number(report, 'resource_reduction'), _number(report, 'resource_reduction_low')
    quality_hi = _number(report, 'quality_gap_upper')
    margin = _number(report, 'quality_margin')
    operational = (saving is not None and saving >= .20 and saving_low is not None and saving_low > 0
                   and quality_hi is not None and margin is not None and quality_hi <= margin
                   and report.get('live_end_to_end_measured') is True
                   and report.get('full_resource_accounting') is True
                   and report.get('operating_point_frozen_on_validation') is True
                   and report.get('absolute_reference_fidelity_reported') is True)
    if operational:
        return {'level': 'CONSEQUENTIAL_SCOPED_RESULT', 'decision_threshold_met': decision,
                'message': 'Registered operational threshold met in the declared scope; not an acceptance guarantee.'}
    if decision:
        return {'level': 'CONFIRMED_DECISION_EFFECT', 'decision_threshold_met': True,
                'message': 'Strong fresh selection result; a resource-importance claim needs additional measurements.'}
    return {'level': 'COMPLETE_BUT_NOT_DECISIVE', 'decision_threshold_met': False,
            'message': 'Report the measured positive/zero/negative results; do not rewrite thresholds after test.'}
