"""Label standardization, sentinel handling, and unit fixes."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import CROP_ALIASES, MONTH_TO_NUM, NUM_TO_MONTH, REGION_ALIASES


def _norm_token(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def standardize_region(value: Any) -> str | float:
    token = _norm_token(value)
    if not token:
        return np.nan
    return REGION_ALIASES.get(token, token.title())


def standardize_crop(value: Any) -> str | float:
    token = _norm_token(value)
    if not token:
        return np.nan
    return CROP_ALIASES.get(token, token)


def standardize_month(value: Any) -> str | float:
    token = _norm_token(value)
    if not token:
        return np.nan
    if token.isdigit():
        num = int(token)
        return NUM_TO_MONTH.get(num, np.nan)
    num = MONTH_TO_NUM.get(token)
    if num is None:
        return np.nan
    return NUM_TO_MONTH[num]


def month_number(value: Any) -> float:
    month = standardize_month(value)
    if month != month:
        return np.nan
    inverse = {v: k for k, v in NUM_TO_MONTH.items()}
    return float(inverse[month])


def replace_sentinels(series: pd.Series, sentinels: tuple[float, ...] = (-999,)) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    return out.replace(list(sentinels), np.nan)


def log_issue(
    rows: list[dict],
    file: str,
    columns: str,
    issue_type: str,
    n_affected: int,
    n_total: int,
    fix: str,
    why: str,
) -> None:
    rows.append(
        {
            "file": file,
            "columns": columns,
            "issue_type": issue_type,
            "n_affected": int(n_affected),
            "pct_affected": round(100.0 * n_affected / max(n_total, 1), 2),
            "fix": fix,
            "why": why,
        }
    )
