"""
cohort_config.py

Single source of truth for the comparator cohort channel list.

Reads include==True rows from cohort/cohort_selection.csv so that any future
cohort change is a one-liner in finalize_cohort.py, not a multi-file edit.

Usage:
    from cohort_config import COHORT, TARGET, POST2022
"""

from pathlib import Path
import pandas as pd

_BASE = Path(__file__).resolve().parent          # features/
_COHORT_CSV = _BASE.parent / "cohort" / "cohort_selection.csv"

TARGET = "TargetCreator"
POST2022 = pd.Timestamp("2022-01-01", tz="UTC")

def _load_cohort() -> list[str]:
    df = pd.read_csv(_COHORT_CSV, encoding="utf-8-sig")
    return df.loc[df["include"] == True, "channelTitle"].tolist()

COHORT: list[str] = _load_cohort()
