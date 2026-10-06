"""Build cleaned, joined, feature-engineered master tables (Deliverable A)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.cleaning import (
    log_issue,
    month_number,
    replace_sentinels,
    standardize_crop,
    standardize_month,
    standardize_region,
)
from src.config import (
    ALLOWED_CROPS,
    ALLOWED_REGIONS,
    APP_ASSETS,
    DATA_PROCESSED,
    DATA_RAW,
    GROWING_SEASON_MONTHS,
    MODEL_FEATURES,
    NUM_TO_MONTH,
    TARGET,
)


@dataclass
class PipelineArtifacts:
    master_train: pd.DataFrame
    master_test: pd.DataFrame
    weather_clean: pd.DataFrame
    prices_clean: pd.DataFrame
    cleaning_log: pd.DataFrame
    label_proof: dict[str, Any]
    join_audit: dict[str, Any]
    join_proof: pd.DataFrame
    feature_table: pd.DataFrame
    integrity_lines: list[str]
    train_stats: dict[str, Any] = field(default_factory=dict)


def _read_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_RAW / "crop_yield_train.csv")
    test = pd.read_csv(DATA_RAW / "crop_yield_leaderboard_test.csv")
    weather = pd.read_csv(DATA_RAW / "regional_weather.csv")
    prices = pd.read_csv(DATA_RAW / "market_prices.csv")
    template = pd.read_csv(DATA_RAW / "submission_template.csv")
    return train, test, weather, prices, template


def _unique_sorted(series: pd.Series) -> list[str]:
    return sorted({str(v) for v in series.dropna().unique()})


def clean_plot_table(df: pd.DataFrame, file_name: str, log_rows: list[dict], stats: dict | None = None) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    fit = stats is None
    local: dict[str, Any] = {} if fit else stats

    before_region = _unique_sorted(out["region"])
    before_crop = _unique_sorted(out["crop_type"])
    out["region"] = out["region"].map(standardize_region)
    out["crop_type"] = out["crop_type"].map(standardize_crop)
    log_issue(
        log_rows, file_name, "region", "inconsistent_labels",
        n, n,
        "strip, lowercase, map aliases, title-case (SNNPR kept)",
        "Region names mixed case, ALLCAPS, and trailing spaces; labels must match weather and price tables.",
    )
    log_issue(
        log_rows, file_name, "crop_type", "inconsistent_labels",
        n, n,
        "strip, lowercase, map Tef→teff",
        "Crop names mixed case, leading/trailing spaces, and Tef vs teff.",
    )
    out.attrs["before_region"] = before_region
    out.attrs["before_crop"] = before_crop
    out.attrs["after_region"] = _unique_sorted(out["region"])
    out.attrs["after_crop"] = _unique_sorted(out["crop_type"])

    pest_n = int((pd.to_numeric(out["pest_disease_flag"], errors="coerce") == -999).sum())
    labor_n = int((pd.to_numeric(out["labor_days_per_ha"], errors="coerce") == -999).sum())
    out["pest_disease_flag"] = replace_sentinels(out["pest_disease_flag"])
    out["labor_days_per_ha"] = replace_sentinels(out["labor_days_per_ha"])
    log_issue(
        log_rows, file_name, "pest_disease_flag", "sentinel_-999",
        pest_n, n, "replace -999 with NaN then impute train mode",
        "-999 is a missing-data code, not a valid binary flag.",
    )
    log_issue(
        log_rows, file_name, "labor_days_per_ha", "sentinel_-999",
        labor_n, n, "replace -999 with NaN then impute train median",
        "Negative labor days are impossible; -999 is a sentinel.",
    )

    farm_hi = out["farm_size_ha"] > 10
    n_farm = int(farm_hi.sum())
    out.loc[farm_hi, "farm_size_ha"] = out.loc[farm_hi, "farm_size_ha"] / 10.0
    log_issue(
        log_rows, file_name, "farm_size_ha", "impossible_outlier_unit",
        n_farm, n, "values > 10 ha divided by 10",
        "Smallholder plots cluster near 1 ha; a clump of 26–60 ha values is a 10× recording error.",
    )

    fert_hi = out["fertilizer_kg_per_ha"] > 200
    n_fert = int(fert_hi.fillna(False).sum())
    out.loc[fert_hi.fillna(False), "fertilizer_kg_per_ha"] = (
        out.loc[fert_hi.fillna(False), "fertilizer_kg_per_ha"] / 10.0
    )
    log_issue(
        log_rows, file_name, "fertilizer_kg_per_ha", "impossible_outlier_unit",
        n_fert, n, "values > 200 kg/ha divided by 10",
        "Cereal rates above 200 kg/ha are rare here; a tail near 900 kg/ha matches a 10× unit error.",
    )

    rain = pd.to_numeric(out["rainfall_mm_season"], errors="coerce")
    if fit:
        local["rain_cap"] = float(rain[rain <= 2000].quantile(0.99))
    rain_hi = rain > 2000
    n_rain = int(rain_hi.fillna(False).sum())
    out.loc[rain_hi.fillna(False), "rainfall_mm_season"] = local["rain_cap"]
    log_issue(
        log_rows, file_name, "rainfall_mm_season", "impossible_outlier",
        n_rain, n, f"cap values > 2000 mm at train 99th percentile of plausible rains ({local['rain_cap']:.1f} mm)",
        "Season totals near 4000 mm are not credible for these regions; the 99th percentile of the rest is ~1400 mm.",
    )

    miss_cols = ["rainfall_mm_season", "fertilizer_kg_per_ha", "soil_quality_index"]
    n_miss = int(out[miss_cols].isna().any(axis=1).sum())
    log_issue(
        log_rows, file_name, ", ".join(miss_cols), "missing_values",
        n_miss, n, "median imputation using train statistics only (Rule 6)",
        "Survey skip/non-response; test has the same holes so we cannot drop rows.",
    )

    out["planting_month"] = out["planting_month"].map(standardize_month)
    out["planting_month_num"] = out["planting_month"].map(month_number)
    return out, local if fit else stats


def clean_weather(df: pd.DataFrame, log_rows: list[dict]) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    before = _unique_sorted(out["region"])
    out["region"] = out["region"].map(standardize_region)
    out["month"] = out["month"].map(standardize_month)
    log_issue(
        log_rows, "regional_weather.csv", "region", "inconsistent_labels",
        n, n, "strip/alias map including ORO, AMH, SOM, TIG, SNNP",
        "Weather regions use abbreviations and mixed case, different from the plot table.",
    )

    n_full_dup = int(out.duplicated().sum())
    out = out.drop_duplicates()
    log_issue(
        log_rows, "regional_weather.csv", "all", "duplicate_rows",
        n_full_dup, n, "drop exact duplicate rows",
        "Repeated export of the same station-month.",
    )

    key = ["region", "year", "month"]
    n_key_dup = int(out.duplicated(key).sum())
    out = (
        out.groupby(key, as_index=False)
        .agg(
            avg_temp_c=("avg_temp_c", "mean"),
            monthly_rainfall_mm=("monthly_rainfall_mm", "mean"),
            extreme_heat_days=("extreme_heat_days", "mean"),
        )
    )
    log_issue(
        log_rows, "regional_weather.csv", "region, year, month", "duplicate_keys",
        n_key_dup, n, "average numeric readings for the same region-year-month",
        "A many-to-one join would explode plot rows if duplicate keys were left in.",
    )

    n_miss = int(out[["avg_temp_c", "monthly_rainfall_mm"]].isna().any(axis=1).sum())
    out["avg_temp_c"] = out["avg_temp_c"].fillna(out.groupby("region")["avg_temp_c"].transform("median"))
    out["monthly_rainfall_mm"] = out["monthly_rainfall_mm"].fillna(
        out.groupby("region")["monthly_rainfall_mm"].transform("median")
    )
    log_issue(
        log_rows, "regional_weather.csv", "avg_temp_c, monthly_rainfall_mm", "missing_values",
        n_miss, n, "fill with region median of the remaining months",
        "A few station-months are blank; region typical climate is a better fill than a global average.",
    )
    out.attrs["before_region"] = before
    out.attrs["after_region"] = _unique_sorted(out["region"])
    return out


def clean_prices(df: pd.DataFrame, log_rows: list[dict], stats: dict | None = None) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    before_region = _unique_sorted(out["region"])
    before_crop = _unique_sorted(out["crop_type"])
    out["region"] = out["region"].map(standardize_region)
    out["crop_type"] = out["crop_type"].map(standardize_crop)
    log_issue(
        log_rows, "market_prices.csv", "crop_type", "inconsistent_labels",
        n, n, "strip, lowercase, map aliases",
        "Crop labels mixed case and padded with spaces; must match plot crop_type.",
    )

    low = out["price_birr_per_quintal"] < 100
    n_low = int(low.fillna(False).sum())
    out.loc[low.fillna(False), "price_birr_per_quintal"] *= 100.0
    log_issue(
        log_rows, "market_prices.csv", "price_birr_per_quintal", "unit_mixup",
        n_low, n, "values < 100 multiplied by 100 (birr/kg → birr/quintal)",
        "1 quintal = 100 kg. Four rows near 34–49 birr sit ~100× below the rest of the same crops (~4000 birr).",
    )

    n_miss = int(out["price_birr_per_quintal"].isna().sum())
    crop_year_med = out.groupby(["crop_type", "year"])["price_birr_per_quintal"].transform("median")
    crop_med = out.groupby("crop_type")["price_birr_per_quintal"].transform("median")
    out["price_birr_per_quintal"] = out["price_birr_per_quintal"].fillna(crop_year_med).fillna(crop_med)
    log_issue(
        log_rows, "market_prices.csv", "price_birr_per_quintal", "missing_values",
        n_miss, n, "impute crop×year median, then crop median",
        "Four region-year cells are blank; neighboring years for the same crop are a fair market proxy.",
    )

    key = ["crop_type", "region", "year"]
    n_dup = int(out.duplicated(key).sum())
    if n_dup:
        out = out.groupby(key, as_index=False)["price_birr_per_quintal"].mean()
    log_issue(
        log_rows, "market_prices.csv", "crop_type, region, year", "duplicate_keys",
        n_dup, n, "average if duplicate keys exist",
        "Guard against a many-to-one join inflating plot rows.",
    )
    out.attrs["before_region"] = before_region
    out.attrs["before_crop"] = before_crop
    out.attrs["after_region"] = _unique_sorted(out["region"])
    out.attrs["after_crop"] = _unique_sorted(out["crop_type"])
    return out


def growing_season_months(plant_month_num: int) -> list[int]:
    return [((int(plant_month_num) - 1 + i) % 12) + 1 for i in range(GROWING_SEASON_MONTHS)]


def season_year_for_month(survey_year: int, plant_month_num: int, month_num: int) -> int:
    """If the window wraps past December, later months belong to the next year."""
    if month_num >= plant_month_num:
        return int(survey_year)
    return int(survey_year) + 1


def _season_keys(plots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in plots[["plot_id", "region", "survey_year", "planting_month_num"]].itertuples(index=False):
        plant = int(rec.planting_month_num)
        for m in growing_season_months(plant):
            rows.append(
                {
                    "plot_id": rec.plot_id,
                    "region": rec.region,
                    "year": season_year_for_month(int(rec.survey_year), plant, m),
                    "month": NUM_TO_MONTH[m],
                    "month_num": m,
                }
            )
    return pd.DataFrame(rows)


def aggregate_weather_for_plots(plots: pd.DataFrame, weather: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = _season_keys(plots)
    merged = keys.merge(weather, on=["region", "year", "month"], how="left")
    agg = merged.groupby("plot_id", as_index=False).agg(
        season_mean_temp_c=("avg_temp_c", "mean"),
        season_total_rainfall_mm=("monthly_rainfall_mm", "sum"),
        season_extreme_heat_days=("extreme_heat_days", "sum"),
        n_season_weather_months=("avg_temp_c", "count"),
    )
    return agg, merged


def engineer_features(plots: pd.DataFrame, weather_agg: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    out = plots.merge(weather_agg, on="plot_id", how="left")
    region_temp = (
        out.groupby("region")["season_mean_temp_c"].transform("mean")
        if "season_mean_temp_c" in out
        else np.nan
    )
    region_rain = out.groupby("region")["season_total_rainfall_mm"].transform("mean")
    out["season_temp_dev_region"] = out["season_mean_temp_c"] - region_temp
    out["season_rain_dev_region"] = out["season_total_rainfall_mm"] - region_rain
    out["fertilizer_x_improved_seed"] = out["fertilizer_kg_per_ha"] * out["improved_seed_used"]
    angle = 2 * np.pi * (out["planting_month_num"] / 12.0)
    out["planting_month_sin"] = np.sin(angle)
    out["planting_month_cos"] = np.cos(angle)
    out["rainfall_station_minus_plot"] = out["season_total_rainfall_mm"] - out["rainfall_mm_season"]
    price_join = prices.rename(columns={"year": "survey_year"})
    before = len(out)
    out = out.merge(price_join, on=["crop_type", "region", "survey_year"], how="left")
    if len(out) != before:
        raise RuntimeError(f"Price join changed row count {before} -> {len(out)}")
    out["revenue_birr_per_ha"] = out[TARGET] * 10 * out["price_birr_per_quintal"] if TARGET in out else np.nan
    return out


def impute_master(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    stats: dict[str, Any] = {"medians": {}, "modes": {}}
    feature_cols = [c for c in MODEL_FEATURES if c in train.columns]
    for col in feature_cols:
        if col in ["region", "crop_type", "planting_month"]:
            mode = train[col].mode(dropna=True)
            stats["modes"][col] = None if mode.empty else str(mode.iloc[0])
        else:
            stats["medians"][col] = float(pd.to_numeric(train[col], errors="coerce").median())

    def apply(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, mode in stats["modes"].items():
            if mode is not None:
                out[col] = out[col].fillna(mode)
        for col, med in stats["medians"].items():
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(med)
        return out

    return apply(train), apply(test), stats


def run_integrity(train: pd.DataFrame, test: pd.DataFrame, n_raw_train: int, n_raw_test: int) -> list[str]:
    lines = []

    def check(name: str, ok: bool, detail: str) -> None:
        status = "PASS" if ok else "FAIL"
        lines.append(f"[{status}] {name}: {detail}")

    check("unique_plot_id_train", train["plot_id"].is_unique, f"n={len(train)}")
    check("unique_plot_id_test", test["plot_id"].is_unique, f"n={len(test)}")
    check("row_count_train", len(train) == n_raw_train, f"{len(train)} vs raw {n_raw_train}")
    check("row_count_test", len(test) == n_raw_test, f"{len(test)} vs raw {n_raw_test}")
    train_feats = [c for c in MODEL_FEATURES if c in train.columns]
    test_feats = [c for c in MODEL_FEATURES if c in test.columns]
    check("same_feature_columns", train_feats == test_feats, f"train={len(train_feats)} test={len(test_feats)}")
    miss_train = int(train[train_feats].isna().sum().sum())
    miss_test = int(test[test_feats].isna().sum().sum())
    check("no_missing_model_features", miss_train == 0 and miss_test == 0, f"train_na={miss_train} test_na={miss_test}")
    ok_region = set(train["region"].unique()) <= set(ALLOWED_REGIONS) and set(test["region"].unique()) <= set(ALLOWED_REGIONS)
    ok_crop = set(train["crop_type"].unique()) <= set(ALLOWED_CROPS) and set(test["crop_type"].unique()) <= set(ALLOWED_CROPS)
    check("allowed_region_crop_labels", ok_region and ok_crop, f"regions={sorted(train['region'].unique())}")
    if TARGET in train:
        check("yield_non_negative", bool((train[TARGET] >= 0).all()), f"min={train[TARGET].min():.3f}")
    check("price_positive", bool((train["price_birr_per_quintal"] > 0).all() and (test["price_birr_per_quintal"] > 0).all()),
          f"train_min={train['price_birr_per_quintal'].min():.1f}")
    return lines


FEATURE_SPECS = [
    {
        "name": "season_mean_temp_c",
        "formula": "mean(avg_temp_c) over planting month + next 3 months",
        "source": "regional_weather.avg_temp_c",
        "why": "Crops have temperature sweet spots; plot data has no temperature.",
    },
    {
        "name": "season_total_rainfall_mm",
        "formula": "sum(monthly_rainfall_mm) over the same 4-month window",
        "source": "regional_weather.monthly_rainfall_mm",
        "why": "Station rainfall is independent of self-reported plot rainfall (Rule 5).",
    },
    {
        "name": "season_extreme_heat_days",
        "formula": "sum(extreme_heat_days) over the window",
        "source": "regional_weather.extreme_heat_days",
        "why": "Heat stress during grain fill reduces yield even if average rain looks fine.",
    },
    {
        "name": "season_temp_dev_region",
        "formula": "season_mean_temp_c - mean(season_mean_temp_c | region)",
        "source": "weather-derived",
        "why": "An unusually hot or cool year relative to the region is more informative than the raw mean.",
    },
    {
        "name": "fertilizer_x_improved_seed",
        "formula": "fertilizer_kg_per_ha * improved_seed_used",
        "source": "plot fertilizer × improved_seed_used",
        "why": "Improved seed often responds more to fertilizer than local varieties (interaction).",
    },
    {
        "name": "planting_month_sin / planting_month_cos",
        "formula": "sin/cos(2π * month_num / 12)",
        "source": "planting_month",
        "why": "Belg vs Meher timing is circular; dummy months ignore that Feb is next to Mar and near Jan.",
    },
    {
        "name": "rainfall_station_minus_plot",
        "formula": "season_total_rainfall_mm - rainfall_mm_season",
        "source": "weather total vs plot rainfall_mm_season",
        "why": "Disagreement may flag recall error or very local storms; residual rain signal for the model.",
    },
]


def build_data_dictionary(train: pd.DataFrame) -> pd.DataFrame:
    desc = {
        "plot_id": ("string", "crop_yield_*.csv", "Unique plot identifier", "raw"),
        "region": ("category", "all three files", "Standardized region", "cleaned labels"),
        "crop_type": ("category", "plot, prices", "Standardized crop", "cleaned labels"),
        "survey_year": ("int", "plot", "Survey year 2021–2024", "raw"),
        "planting_month": ("category", "plot", "Planting month", "standardized to Jan–Dec"),
        "planting_month_num": ("int", "plot", "Planting month 1–12", "from planting_month"),
        "altitude_m": ("float", "plot", "Elevation meters", "raw"),
        "rainfall_mm_season": ("float", "plot", "Self-reported season rainfall mm", "sentinel/outlier cleaned, imputed"),
        "farm_size_ha": ("float", "plot", "Farm size hectares", "10× outliers divided by 10"),
        "fertilizer_kg_per_ha": ("float", "plot", "Fertilizer kg/ha", "10× outliers divided by 10, imputed"),
        "improved_seed_used": ("int", "plot", "Certified seed flag", "raw 0/1"),
        "pest_disease_flag": ("float", "plot", "Pest/disease flag", "-999 → NaN → imputed"),
        "soil_quality_index": ("float", "plot", "Soil quality 0–1", "imputed"),
        "labor_days_per_ha": ("float", "plot", "Labor days per ha", "-999 → NaN → imputed"),
        "distance_to_market_km": ("float", "plot", "Distance to market km", "raw"),
        TARGET: ("float", "plot train only", "Yield tons/ha", "raw target"),
        "season_mean_temp_c": ("float", "weather", "Growing-season mean temperature", "join aggregate"),
        "season_total_rainfall_mm": ("float", "weather", "Growing-season rainfall total", "join aggregate"),
        "season_extreme_heat_days": ("float", "weather", "Growing-season extreme heat days", "join aggregate"),
        "n_season_weather_months": ("float", "weather", "Months observed in the 4-month window", "join audit"),
        "season_temp_dev_region": ("float", "weather", "Season temp minus region mean", "engineered"),
        "season_rain_dev_region": ("float", "weather", "Season rain minus region mean", "engineered"),
        "fertilizer_x_improved_seed": ("float", "plot", "Fertilizer × improved seed", "engineered"),
        "planting_month_sin": ("float", "plot", "Cyclical planting month sine", "engineered"),
        "planting_month_cos": ("float", "plot", "Cyclical planting month cosine", "engineered"),
        "rainfall_station_minus_plot": ("float", "weather+plot", "Station minus plot rainfall", "engineered"),
        "price_birr_per_quintal": ("float", "prices", "Market price (analysis/demo only, not a model feature)", "unit-corrected"),
        "revenue_birr_per_ha": ("float", "plot×price", "yield × 10 × price (train analysis)", "derived"),
    }
    rows = []
    for col in train.columns:
        typ, src, d, derived = desc.get(col, (str(train[col].dtype), "unknown", col, "passthrough"))
        rows.append({"column": col, "type": typ, "source_file": src, "description": d, "how_derived": derived})
    return pd.DataFrame(rows)


def run_pipeline() -> PipelineArtifacts:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    APP_ASSETS.mkdir(parents=True, exist_ok=True)
    log_rows: list[dict] = []

    train_raw, test_raw, weather_raw, prices_raw, _template = _read_raw()
    n_raw_train, n_raw_test = len(train_raw), len(test_raw)

    label_proof = {
        "plot_train_region_before": _unique_sorted(train_raw["region"]),
        "plot_train_crop_before": _unique_sorted(train_raw["crop_type"]),
        "plot_test_region_before": _unique_sorted(test_raw["region"]),
        "plot_test_crop_before": _unique_sorted(test_raw["crop_type"]),
        "weather_region_before": _unique_sorted(weather_raw["region"]),
        "price_region_before": _unique_sorted(prices_raw["region"]),
        "price_crop_before": _unique_sorted(prices_raw["crop_type"]),
    }

    train, plot_stats = clean_plot_table(train_raw, "crop_yield_train.csv", log_rows, None)
    test, _ = clean_plot_table(test_raw, "crop_yield_leaderboard_test.csv", log_rows, plot_stats)
    weather = clean_weather(weather_raw, log_rows)
    prices = clean_prices(prices_raw, log_rows)

    label_proof.update(
        {
            "plot_train_region_after": _unique_sorted(train["region"]),
            "plot_train_crop_after": _unique_sorted(train["crop_type"]),
            "weather_region_after": _unique_sorted(weather["region"]),
            "price_region_after": _unique_sorted(prices["region"]),
            "price_crop_after": _unique_sorted(prices["crop_type"]),
        }
    )

    w_train, w_merged_train = aggregate_weather_for_plots(train, weather)
    w_test, w_merged_test = aggregate_weather_for_plots(test, weather)

    n_before_w = len(train)
    train_j = engineer_features(train, w_train, prices)
    test_j = engineer_features(test, w_test, prices)
    assert len(train_j) == n_before_w
    assert len(test_j) == n_raw_test

    unmatched_w_train = int((train_j["n_season_weather_months"] == 0).sum())
    unmatched_w_test = int((test_j["n_season_weather_months"] == 0).sum())
    incomplete_train = int((train_j["n_season_weather_months"] < GROWING_SEASON_MONTHS).sum())
    incomplete_test = int((test_j["n_season_weather_months"] < GROWING_SEASON_MONTHS).sum())
    unmatched_p_train = int(train_j["price_birr_per_quintal"].isna().sum())
    unmatched_p_test = int(test_j["price_birr_per_quintal"].isna().sum())

    join_audit = {
        "weather": {
            "left_table": "plots (train/test)",
            "right_table": "regional_weather aggregated to season",
            "join_type": "left many-to-one on plot_id after expanding planting_month+3",
            "expected_cardinality": "many plots to one season-weather row",
            "train_before": n_raw_train,
            "train_after": len(train_j),
            "test_before": n_raw_test,
            "test_after": len(test_j),
            "plots_with_zero_weather_months_train": unmatched_w_train,
            "plots_with_zero_weather_months_test": unmatched_w_test,
            "plots_with_incomplete_window_train": incomplete_train,
            "plots_with_incomplete_window_test": incomplete_test,
            "handling": "Aggregate available months; impute remaining weather features from train medians.",
            "match_rate_train_any_month": round(1 - unmatched_w_train / n_raw_train, 4),
            "match_rate_train_full_window": round(1 - incomplete_train / n_raw_train, 4),
        },
        "price": {
            "left_table": "plots",
            "right_table": "market_prices",
            "join_keys": "region, crop_type, survey_year",
            "join_type": "left many-to-one",
            "train_before": n_raw_train,
            "train_after": len(train_j),
            "unmatched_train": unmatched_p_train,
            "unmatched_test": unmatched_p_test,
            "handling": "Impute remaining prices from crop×year then crop median (already applied in price clean; leftover filled later).",
            "match_rate_train": round(1 - unmatched_p_train / n_raw_train, 4),
        },
        "growing_season_rule": "planting month plus the following three months, wrapping the calendar if needed",
        "why_left_is_plots": "Scoring is plot-level; plots are the grain of the prediction task, so plots stay on the left of every join.",
    }

    proof_ids = [train_j["plot_id"].iloc[i] for i in (0, len(train_j) // 3, -1)]
    proof_rows = []
    for pid in proof_ids:
        window = w_merged_train[w_merged_train["plot_id"] == pid].copy()
        plot = train_j[train_j["plot_id"] == pid].iloc[0]
        for _, wr in window.iterrows():
            proof_rows.append(
                {
                    "plot_id": pid,
                    "region": plot["region"],
                    "crop_type": plot["crop_type"],
                    "survey_year": int(plot["survey_year"]),
                    "planting_month": plot["planting_month"],
                    "weather_year": wr["year"],
                    "weather_month": wr["month"],
                    "avg_temp_c": wr["avg_temp_c"],
                    "monthly_rainfall_mm": wr["monthly_rainfall_mm"],
                    "extreme_heat_days": wr["extreme_heat_days"],
                    "season_mean_temp_c": plot["season_mean_temp_c"],
                    "season_total_rainfall_mm": plot["season_total_rainfall_mm"],
                    "season_extreme_heat_days": plot["season_extreme_heat_days"],
                    "n_season_weather_months": plot["n_season_weather_months"],
                }
            )
    join_proof = pd.DataFrame(proof_rows)

    train_m, test_m, impute_stats = impute_master(train_j, test_j)
    if TARGET in train_m:
        train_m["revenue_birr_per_ha"] = train_m[TARGET] * 10 * train_m["price_birr_per_quintal"]

    integrity = run_integrity(train_m, test_m, n_raw_train, n_raw_test)
    for line in integrity:
        if line.startswith("[FAIL]"):
            raise AssertionError(line)

    feature_table = pd.DataFrame(FEATURE_SPECS)
    dictionary = build_data_dictionary(train_m)

    drop_for_test = [c for c in [TARGET, "revenue_birr_per_ha"] if c in test_m.columns]
    test_out = test_m.drop(columns=drop_for_test, errors="ignore")

    train_m.to_csv(DATA_PROCESSED / "master_train.csv", index=False)
    test_out.to_csv(DATA_PROCESSED / "master_test.csv", index=False)
    dictionary.to_csv(DATA_PROCESSED / "data_dictionary_master.csv", index=False)
    pd.DataFrame(log_rows).to_csv(DATA_PROCESSED / "cleaning_log.csv", index=False)
    join_proof.to_csv(DATA_PROCESSED / "join_proof.csv", index=False)
    weather.to_csv(APP_ASSETS / "weather_clean.csv", index=False)
    prices.to_csv(APP_ASSETS / "prices_clean.csv", index=False)
    weather.to_csv(DATA_PROCESSED / "weather_clean.csv", index=False)
    prices.to_csv(DATA_PROCESSED / "prices_clean.csv", index=False)

    stats_out = {"plot": plot_stats, "impute": impute_stats}
    (DATA_PROCESSED / "train_stats.json").write_text(json.dumps(stats_out, indent=2), encoding="utf-8")

    return PipelineArtifacts(
        master_train=train_m,
        master_test=test_out,
        weather_clean=weather,
        prices_clean=prices,
        cleaning_log=pd.DataFrame(log_rows),
        label_proof=label_proof,
        join_audit=join_audit,
        join_proof=join_proof,
        feature_table=feature_table,
        integrity_lines=integrity,
        train_stats=stats_out,
    )


if __name__ == "__main__":
    art = run_pipeline()
    print(art.cleaning_log.to_string(index=False))
    print("\n".join(art.integrity_lines))
    print(art.master_train.shape, art.master_test.shape)
