from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernel_isqed.io import CANONICAL_DIR
from kernel_isqed.run_keys import build_expected_run_keys


def main() -> None:
    split = pd.read_parquet(CANONICAL_DIR / "split_manifest.parquet")
    state_path = CANONICAL_DIR / "serialized_states.parquet"
    states = pd.read_parquet(state_path) if state_path.is_file() else None
    frame = build_expected_run_keys(split, states)
    print(frame.groupby("key_family").size().rename("expected_keys").to_string())
    print("saved:", CANONICAL_DIR / "expected_run_keys.parquet")


if __name__ == "__main__":
    main()
