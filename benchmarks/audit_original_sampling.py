"""Reproduce sampling observations from published metadata, without response losses."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    here = Path(__file__).resolve().parent
    source = here.parent / "reference_implementation/data/canonical/split_manifest.parquet"
    frame = pd.read_parquet(source, columns=["task_id", "item_id", "domain", "gold", "split"])
    domain_rows, label_rows = [], []
    mixtures = {}
    for task, task_frame in frame.groupby("task_id"):
        domains = sorted(task_frame.domain.unique())
        labels = sorted(task_frame.gold.unique())
        for split, part in task_frame.groupby("split"):
            dc, lc = part.domain.value_counts(), part.gold.value_counts()
            common = {"task_id": task, "split": split, "partition_items": len(part)}
            domain_rows.extend({**common, "domain": domain, "count": int(dc.get(domain, 0)),
                                "fraction": float(dc.get(domain, 0) / len(part))} for domain in domains)
            label_rows.extend({**common, "gold": int(label), "count": int(lc.get(label, 0)),
                               "fraction": float(lc.get(label, 0) / len(part))} for label in labels)
        val = task_frame[task_frame.split == "validation"].domain.value_counts(normalize=True)
        test = task_frame[task_frame.split == "iid_test"].domain.value_counts(normalize=True)
        mixtures[task] = float(0.5 * val.subtract(test, fill_value=0).abs().sum())
    mmlu_test = frame[(frame.task_id == "mmlu_pro") & (frame.split == "iid_test")]
    mmlu_counts = mmlu_test.domain.value_counts().sort_index()
    frequent = mmlu_counts[mmlu_counts > 4]
    remaining = mmlu_counts[mmlu_counts == 4]
    hella = frame[frame.task_id == "hellaswag"]
    hella_test = hella[hella.split == "iid_test"]
    sparse_domains = sorted(hella_test.domain.value_counts().loc[lambda x: x == 4].index)
    hella_val = hella[hella.split == "validation"]
    sparse_test_count = int(hella_test.domain.isin(sparse_domains).sum())
    sparse_val_count = int(hella_val.domain.isin(sparse_domains).sum())
    report = {
        "scope": "Audit of published split metadata only; no new replacement-performance claim and no response losses read.",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "mmlu_pro_iid_test": {
            "total_items": len(mmlu_test),
            "domains_with_more_than_four_items": frequent.to_dict(),
            "items_in_those_six_domains": int(frequent.sum()),
            "fraction_in_those_six_domains": float(frequent.sum() / len(mmlu_test)),
            "remaining_domains": remaining.to_dict(),
            "items_in_remaining_seven_domains": int(remaining.sum()),
        },
        "hellaswag_validation_test_mixture": {
            "domains_with_exactly_four_test_items": sparse_domains,
            "number_of_those_domains": len(sparse_domains),
            "test_items_in_those_domains": sparse_test_count,
            "test_total": len(hella_test),
            "test_fraction": sparse_test_count / len(hella_test),
            "validation_items_in_those_same_domains": sparse_val_count,
            "validation_total": len(hella_val),
            "validation_fraction": sparse_val_count / len(hella_val),
            "domain_mixture_total_variation": mixtures["hellaswag"],
        },
        "validation_test_domain_mixture_total_variation_by_task": mixtures,
    }
    assert len(frequent) == 6 and frequent.sum() == 996 and len(remaining) == 7
    assert len(sparse_domains) == 66 and sparse_test_count == 264
    (here / "original_sampling_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(domain_rows).to_csv(here / "original_sampling_domain_counts.csv", index=False)
    pd.DataFrame(label_rows).to_csv(here / "original_sampling_label_counts.csv", index=False)
    print(json.dumps({
        "mmlu_test_six_domain_items": int(frequent.sum()),
        "hella_test_minimum_count_domain_fraction": sparse_test_count / len(hella_test),
        "hella_validation_same_domain_fraction": sparse_val_count / len(hella_val),
        "domain_mixture_tv": mixtures,
    }, indent=2))


if __name__ == "__main__":
    main()
