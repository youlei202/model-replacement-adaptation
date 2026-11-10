from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
STATES_DIR = RESULTS_DIR / "states"
CANONICAL_DIR = PROJECT_ROOT / "data" / "canonical"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(payload.encode("utf-8"))


def write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError("table path must end in .parquet or .csv")
    return path


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("table path must end in .parquet or .csv")


def save_experiment(df: pd.DataFrame, filename: str, canonical_name: Optional[str] = None) -> Path:
    output = write_table(df, TABLES_DIR / filename)
    if canonical_name:
        write_table(df, CANONICAL_DIR / canonical_name)
    return output


def append_table(rows: Iterable[Dict[str, Any]], path: Path) -> Path:
    incoming = pd.DataFrame(list(rows))
    if path.exists():
        current = read_table(path)
        incoming = pd.concat([current, incoming], ignore_index=True)
    return write_table(incoming, path)
