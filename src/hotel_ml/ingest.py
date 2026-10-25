"""Read the two raw hotel extracts exactly as delivered.

Everything is read as text on purpose: the raw files contain fixed-width padding
("BB       "), literal "NULL" strings and mixed types, and we want the cleaning
step to see (and log) all of it rather than letting the CSV parser guess.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def to_snake(name: str) -> str:
    name = name.replace("ADR", "Adr")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def load_raw(raw_dir: Path, files: dict[str, str]) -> pd.DataFrame:
    frames = []
    for hotel, fname in files.items():
        df = pd.read_csv(Path(raw_dir) / fname, dtype=str, keep_default_na=False)
        df.insert(0, "hotel", hotel)
        frames.append(df)
    cols = [set(f.columns) for f in frames]
    if any(c != cols[0] for c in cols):
        raise ValueError("Raw files have different schemas")
    df = pd.concat(frames, ignore_index=True)
    df.columns = [to_snake(c) for c in df.columns]
    return df
