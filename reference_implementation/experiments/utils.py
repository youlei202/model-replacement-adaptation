from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from kernel_isqed.io import CANONICAL_DIR


def common_parser(description: str, include_fast: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--responses", type=Path, default=CANONICAL_DIR / "matched_responses.parquet")
    if include_fast:
        parser.add_argument("--fast", action="store_true")
    return parser


def tag_run_mode(df: pd.DataFrame, fast: bool = False) -> pd.DataFrame:
    result = df.copy()
    result["run_mode"] = "fast" if fast else "full"
    return result
