"""
Reusable data-cleaning pipeline primitives.

The idea: split cleaning into small, testable steps (validate -> impute ->
normalize text -> handle outliers -> type cast) and chain them together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Configuration layer
# ---------------------------------------------------------------------------


@dataclass
class ColumnRule:
    """Rules applied to a single column."""

    required: bool = False #是否必须列
    dtype: Optional[str] = None #目标数据类型
    default: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    categories: Optional[Iterable[str]] = None
    strip_text: bool = True
    title_case: bool = False


@dataclass
class CleaningConfig:
    """High-level knobs for the pipeline."""

    column_rules: Dict[str, ColumnRule] = field(default_factory=dict)
    drop_duplicates: bool = True #是否去重
    duplicate_subset: Optional[List[str]] = None
    outlier_zscore_threshold: float = 3.5
    date_columns: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. Helper functions (each targets one task)
# ---------------------------------------------------------------------------


def validate_columns(df: pd.DataFrame, config: CleaningConfig) -> None:
    missing = [
        name for name, rule in config.column_rules.items()
        if rule.required and name not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def fill_defaults(df: pd.DataFrame, config: CleaningConfig) -> pd.DataFrame:
    result = df.copy()
    for column, rule in config.column_rules.items():
        if column not in result.columns or rule.default is None:
            continue
        result[column] = result[column].fillna(rule.default)
    return result


def cast_types(df: pd.DataFrame, config: CleaningConfig) -> pd.DataFrame:
    result = df.copy()
    for column, rule in config.column_rules.items():
        if column not in result.columns or not rule.dtype:
            continue
        result[column] = result[column].astype(rule.dtype, errors="ignore")
    return result


def clip_numeric_ranges(df: pd.DataFrame, config: CleaningConfig) -> pd.DataFrame:
    result = df.copy()
    for column, rule in config.column_rules.items():
        if column not in result.columns:
            continue
        series = result[column]
        if np.issubdtype(series.dtype, np.number):
            if rule.min_value is not None:
                series = np.maximum(series, rule.min_value)
            if rule.max_value is not None:
                series = np.minimum(series, rule.max_value)
            result[column] = series
    return result


def normalize_text_columns(df: pd.DataFrame, config: CleaningConfig) -> pd.DataFrame:
    result = df.copy()
    for column, rule in config.column_rules.items():
        if column not in result.columns:
            continue
        if not pd.api.types.is_string_dtype(result[column]):
            continue
        series = result[column].astype(str)
        if rule.strip_text:
            series = series.str.strip()
        if rule.title_case:
            series = series.str.title()
        result[column] = series
    return result


def prune_outliers_zscore(df: pd.DataFrame, config: CleaningConfig) -> pd.DataFrame:
    if not np.isfinite(config.outlier_zscore_threshold):
        return df
    threshold = config.outlier_zscore_threshold
    result = df.copy()
    for column in result.select_dtypes(include=[np.number]).columns:
        series = result[column]
        z_scores = (series - series.mean()) / series.std(ddof=0)
        mask = z_scores.abs() > threshold
        if mask.any():
            result.loc[mask, column] = np.nan
    return result


def parse_dates(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
    result = df.copy()
    for column in date_columns:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


# ---------------------------------------------------------------------------
# 3. Pipeline orchestrator
# ---------------------------------------------------------------------------


class CleaningPipeline:
    """
    Simple orchestrator: holds the config and runs each step in sequence.
    You can comment out steps or add new ones without touching the rest.
    """

    def __init__(self, config: CleaningConfig):
        self.config = config

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        validate_columns(df, self.config)
        cleaned = df.copy()

        cleaned = fill_defaults(cleaned, self.config)
        cleaned = normalize_text_columns(cleaned, self.config)
        cleaned = cast_types(cleaned, self.config)
        cleaned = parse_dates(cleaned, self.config.date_columns)
        cleaned = clip_numeric_ranges(cleaned, self.config)
        cleaned = prune_outliers_zscore(cleaned, self.config)

        if self.config.drop_duplicates:
            cleaned = cleaned.drop_duplicates(subset=self.config.duplicate_subset)

        cleaned["data_state"] = "cleaned"
        return cleaned


# ---------------------------------------------------------------------------
# 4. Example usage (replace with tests later)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    raw = pd.DataFrame(
        {
            "姓名": [" 张三 ", None, "LISA"],
            "年龄": [23, None, 140],
            "工资": [5000, 8000, 120000],
            "入职日期": ["2022-01-10", "not a date", "2020/05/03"],
        }
    )

    config = CleaningConfig(
        column_rules={
            "姓名": ColumnRule(required=True, strip_text=True, title_case=True, default="未知"),
            "年龄": ColumnRule(required=True, dtype="float", min_value=0, max_value=120, default=30),
            "工资": ColumnRule(dtype="float", min_value=0),
            "入职日期": ColumnRule(dtype="string"),
        },
        date_columns=["入职日期"],
    )

    pipeline = CleaningPipeline(config)
    cleaned_df = pipeline.run(raw)
    print(cleaned_df)


