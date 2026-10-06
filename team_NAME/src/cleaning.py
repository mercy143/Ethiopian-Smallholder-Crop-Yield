"""Utilities for cleaning the crop-yield challenge data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

REGION_ALIASES = {
    "oromia": "Oromia",
    "oro": "Oromia",
    "amhara": "Amhara",
    "amh": "Amhara",
    "snnpr": "SNNPR",
    "snnp": "SNNPR",
    "tigray": "Tigray",
    "tig": "Tigray",
    "somali": "Somali",
    "som": "Somali",
}

CROP_ALIASES = {
    "teff": "teff",
    "tef": "teff",
    "wheat": "wheat",
    "maize": "maize",
    "corn": "maize",
    "sorghum": "sorghum",
    "barley": "barley",
}

MONTH_ALIASES = {
    "jan": "Jan", "january": "Jan",
    "feb": "Feb", "february": "Feb",
    "mar": "Mar", "march": "Mar",
    "apr": "Apr", "april": "Apr",
    "may": "May",
    "jun": "Jun", "june": "Jun",
    "jul": "Jul", "july": "Jul",
    "aug": "Aug", "august": "Aug",
    "sep": "Sep", "sept": "Sep", "september": "Sep",
    "oct": "Oct", "october": "Oct",
    "nov": "Nov", "november": "Nov",
    "dec": "Dec", "december": "Dec",
}

MONTH_TO_NUM = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def _clean_token(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def standardize_region(value: Any) -> str:
    token = _clean_token(value)
    if not token:
        return np.nan
    return REGION_ALIASES.get(token, token.title())


def standardize_crop(value: Any) -> str:
    token = _clean_token(value)
    if not token:
        return np.nan
    return CROP_ALIASES.get(token, token)


def standardize_month(value: Any) -> str:
    token = _clean_token(value)
    if not token:
        return np.nan
    if token.isdigit():
        month_num = int(token)
        month_name = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][month_num - 1]
        return month_name
    return MONTH_ALIASES.get(token, token.title())


def as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def fill_missing_by_region(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = out[col].fillna(out.groupby("region")[col].transform("median"))
    return out


def clean_plot_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["region"] = out["region"].map(standardize_region)
    out["crop_type"] = out["crop_type"].map(standardize_crop)
    out["planting_month"] = out["planting_month"].map(standardize_month)
    out["planting_month_num"] = out["planting_month"].map(lambda m: MONTH_TO_NUM.get(m, np.nan))

    numeric_cols = [
        "altitude_m",
        "rainfall_mm_season",
        "farm_size_ha",
        "fertilizer_kg_per_ha",
        "improved_seed_used",
        "pest_disease_flag",
        "soil_quality_index",
        "labor_days_per_ha",
        "distance_to_market_km",
        "yield_tons_per_ha",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = as_numeric(out[col])

    out["pest_disease_flag"] = out["pest_disease_flag"].replace(-999, np.nan)
    out["labor_days_per_ha"] = out["labor_days_per_ha"].replace(-999, np.nan)

    farm_mask = out["farm_size_ha"] > 10
    out.loc[farm_mask, "farm_size_ha"] = out.loc[farm_mask, "farm_size_ha"] / 10

    fert_mask = out["fertilizer_kg_per_ha"] > 200
    out.loc[fert_mask, "fertilizer_kg_per_ha"] = out.loc[fert_mask, "fertilizer_kg_per_ha"] / 10

    out["rainfall_mm_season"] = out["rainfall_mm_season"].clip(upper=2000)
    out["fertilizer_kg_per_ha"] = out["fertilizer_kg_per_ha"].clip(lower=0)
    out["farm_size_ha"] = out["farm_size_ha"].clip(lower=0)
    out["distance_to_market_km"] = out["distance_to_market_km"].clip(lower=0)

    return out


def clean_weather_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["region"] = out["region"].map(standardize_region)
    out["month"] = out["month"].map(standardize_month)
    out = out.drop_duplicates()

    numeric_cols = ["avg_temp_c", "monthly_rainfall_mm", "extreme_heat_days"]
    for col in numeric_cols:
        out[col] = as_numeric(out[col])

    out["avg_temp_c"] = out.groupby(["region", "month"])["avg_temp_c"].transform(
        lambda s: s.fillna(s.median())
    )
    out["monthly_rainfall_mm"] = out.groupby(["region", "month"])["monthly_rainfall_mm"].transform(
        lambda s: s.fillna(s.median())
    )
    out["extreme_heat_days"] = out["extreme_heat_days"].fillna(0)
    return out


def clean_price_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["crop_type"] = out["crop_type"].map(standardize_crop)
    out["region"] = out["region"].map(standardize_region)
    out["price_birr_per_quintal"] = as_numeric(out["price_birr_per_quintal"])

    cheap_mask = out["price_birr_per_quintal"] < 100
    out.loc[cheap_mask, "price_birr_per_quintal"] = out.loc[cheap_mask, "price_birr_per_quintal"] * 100

    out["price_birr_per_quintal"] = out.groupby(["crop_type", "year"])["price_birr_per_quintal"].transform(
        lambda s: s.fillna(s.median())
    )
    out = out.drop_duplicates(subset=["crop_type", "region", "year"], keep="first")
    return out


def add_weather_features(plot_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if plot_df.empty:
        result = plot_df.copy()
        for col in [
            "season_mean_temp_c",
            "season_total_rainfall_mm",
            "season_extreme_heat_days",
            "n_season_weather_months",
            "season_temp_dev_region",
            "season_rain_dev_region",
            "fertilizer_x_improved_seed",
            "planting_month_sin",
            "planting_month_cos",
            "rainfall_station_minus_plot",
        ]:
            result[col] = pd.Series(dtype="float64")
        return result

    rows = []

    for _, row in plot_df[["plot_id", "region", "survey_year", "planting_month_num"]].drop_duplicates().iterrows():
        plant_month = int(row["planting_month_num"])
        season_months = []
        for i in range(4):
            month_num = ((plant_month - 1 + i) % 12) + 1
            season_months.append(month_order[month_num - 1])

        year = int(row["survey_year"])
        season = []
        for month in season_months:
            weather_match = weather_df[
                (weather_df["region"] == row["region"]) &
                (weather_df["year"] == year) &
                (weather_df["month"] == month)
            ]
            if not weather_match.empty:
                season.append(weather_match.iloc[0].to_dict())

        if not season:
            seasonal = {
                "plot_id": row["plot_id"],
                "season_mean_temp_c": np.nan,
                "season_total_rainfall_mm": np.nan,
                "season_extreme_heat_days": np.nan,
                "n_season_weather_months": 0,
            }
        else:
            seasonal = {
                "plot_id": row["plot_id"],
                "season_mean_temp_c": float(np.mean([m["avg_temp_c"] for m in season])),
                "season_total_rainfall_mm": float(np.sum([m["monthly_rainfall_mm"] for m in season])),
                "season_extreme_heat_days": float(np.sum([m["extreme_heat_days"] for m in season])),
                "n_season_weather_months": len(season),
            }
        rows.append(seasonal)

    weather_agg = pd.DataFrame(rows)
    merged = plot_df.merge(weather_agg, on="plot_id", how="left")

    region_temp = merged.groupby("region")["season_mean_temp_c"].transform("mean")
    region_rain = merged.groupby("region")["season_total_rainfall_mm"].transform("mean")
    merged["season_temp_dev_region"] = merged["season_mean_temp_c"] - region_temp
    merged["season_rain_dev_region"] = merged["season_total_rainfall_mm"] - region_rain
    merged["fertilizer_x_improved_seed"] = merged["fertilizer_kg_per_ha"] * merged["improved_seed_used"]
    merged["planting_month_sin"] = np.sin(2 * np.pi * merged["planting_month_num"] / 12)
    merged["planting_month_cos"] = np.cos(2 * np.pi * merged["planting_month_num"] / 12)
    merged["rainfall_station_minus_plot"] = merged["season_total_rainfall_mm"] - merged["rainfall_mm_season"]
    return merged


def build_master_tables(train_df: pd.DataFrame, test_df: pd.DataFrame, weather_df: pd.DataFrame, price_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = add_weather_features(train_df, weather_df)
    test = add_weather_features(test_df, weather_df)

    price_train = price_df.rename(columns={"year": "survey_year"})
    train = train.merge(price_train, on=["crop_type", "region", "survey_year"], how="left")
    test = test.merge(price_train, on=["crop_type", "region", "survey_year"], how="left")

    train["revenue_birr_per_ha"] = train["yield_tons_per_ha"] * 10 * train["price_birr_per_quintal"]
    return train, test


def save_processed_data(train: pd.DataFrame, test: pd.DataFrame, root: Path) -> None:
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    train.to_csv(processed / "master_train.csv", index=False)
    test.drop(columns=["yield_tons_per_ha"], errors="ignore").to_csv(processed / "master_test.csv", index=False)

    data_dictionary = pd.DataFrame(
        {
            "column": train.columns.tolist(),
            "type": [str(train[col].dtype) for col in train.columns],
            "source": ["raw plot" if col in {"plot_id", "region", "crop_type", "survey_year", "planting_month", "altitude_m", "rainfall_mm_season", "farm_size_ha", "fertilizer_kg_per_ha", "improved_seed_used", "pest_disease_flag", "soil_quality_index", "labor_days_per_ha", "distance_to_market_km", "yield_tons_per_ha"} else "engineered" for col in train.columns],
            "description": [
                "Unique plot identifier",
                "Standardized region label",
                "Standardized crop label",
                "Survey year",
                "Planting month",
                "Altitude above sea level",
                "Self-reported seasonal rainfall",
                "Farm size in hectares",
                "Fertilizer applied per hectare",
                "Improved seed flag",
                "Pest/disease indicator",
                "Soil quality score",
                "Labor days per hectare",
                "Distance to nearest market",
                "Target yield in tons per hectare",
                *["Weather-derived growing-season mean temperature"] * 8,
            ][: len(train.columns)],
        }
    )
    data_dictionary.to_csv(processed / "data_dictionary_master.csv", index=False)
