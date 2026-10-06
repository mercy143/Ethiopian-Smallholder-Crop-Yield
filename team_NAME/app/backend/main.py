from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "final_model.joblib"
WEATHER_PATH = ROOT / "app" / "assets" / "weather_clean.csv"
PRICE_PATH = ROOT / "app" / "assets" / "prices_clean.csv"
PROCESSED_TRAIN_PATH = ROOT / "data" / "processed" / "master_train.csv"
RAW_WEATHER = ROOT / "data" / "raw" / "regional_weather.csv"
RAW_PRICE = ROOT / "data" / "raw" / "market_prices.csv"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_TO_NUM = {m: i for i, m in enumerate(MONTHS, start=1)}
MODEL_FILE_MAP = {
    "Random Forest": "random_forest.joblib",
    "Gradient Boosting": "gradient_boosting.joblib",
    "Extra Trees": "extra_trees.joblib",
    "Linear Regression": "linear_regression.joblib",
}
MODEL_ALIASES = {name.lower(): name for name in MODEL_FILE_MAP}

app = FastAPI(title="Crop Yield Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionInput(BaseModel):
    region: str
    crop_type: str
    survey_year: int
    planting_month: str
    model_name: str = "Random Forest"
    altitude_m: float
    rainfall_mm_season: float
    farm_size_ha: float
    fertilizer_kg_per_ha: float
    improved_seed_used: int
    pest_disease_flag: int
    soil_quality_index: float
    labor_days_per_ha: float
    distance_to_market_km: float


MODEL_CACHE: dict[str, Any] = {}
for model_name, filename in MODEL_FILE_MAP.items():
    model_path = ROOT / "models" / filename
    if model_path.exists():
        MODEL_CACHE[model_name] = joblib.load(model_path)
if "Random Forest" not in MODEL_CACHE and MODEL_PATH.exists():
    MODEL_CACHE["Random Forest"] = joblib.load(MODEL_PATH)


def _resolve_model(model_name: str | None) -> tuple[Any, str]:
    selected_model = (model_name or "Random Forest").strip()
    key = MODEL_ALIASES.get(selected_model.lower(), selected_model)
    if key in MODEL_CACHE:
        return MODEL_CACHE[key], key
    if "Random Forest" in MODEL_CACHE:
        return MODEL_CACHE["Random Forest"], "Random Forest"
    raise ValueError(f"Model '{selected_model}' is not available")


def _standardize_label(value: str, aliases: dict[str, str]) -> str:
    token = str(value).strip().lower()
    return aliases.get(token, token.title())


def _load_weather() -> pd.DataFrame:
    if WEATHER_PATH.exists():
        return pd.read_csv(WEATHER_PATH)
    df = pd.read_csv(RAW_WEATHER)
    df["region"] = df["region"].map(lambda x: _standardize_label(str(x), {"oromia": "Oromia", "oro": "Oromia", "amhara": "Amhara", "amh": "Amhara", "snnpr": "SNNPR", "snnp": "SNNPR", "tigray": "Tigray", "tig": "Tigray", "somali": "Somali", "som": "Somali"}))
    df["month"] = df["month"].map(lambda x: {"jan": "Jan", "january": "Jan", "feb": "Feb", "february": "Feb", "mar": "Mar", "march": "Mar", "apr": "Apr", "april": "Apr", "may": "May", "jun": "Jun", "june": "Jun", "jul": "Jul", "july": "Jul", "aug": "Aug", "august": "Aug", "sep": "Sep", "sept": "Sep", "september": "Sep", "oct": "Oct", "october": "Oct", "nov": "Nov", "november": "Nov", "dec": "Dec", "december": "Dec"}.get(str(x).strip().lower(), str(x).strip().title()))
    return df


def _load_prices() -> pd.DataFrame:
    if PRICE_PATH.exists():
        return pd.read_csv(PRICE_PATH)
    df = pd.read_csv(RAW_PRICE)
    df["crop_type"] = df["crop_type"].map(lambda x: {"teff": "teff", "tef": "teff", "wheat": "wheat", "maize": "maize", "corn": "maize", "sorghum": "sorghum", "barley": "barley"}.get(str(x).strip().lower(), str(x).strip().lower()))
    df["region"] = df["region"].map(lambda x: _standardize_label(str(x), {"oromia": "Oromia", "oro": "Oromia", "amhara": "Amhara", "amh": "Amhara", "snnpr": "SNNPR", "snnp": "SNNPR", "tigray": "Tigray", "tig": "Tigray", "somali": "Somali", "som": "Somali"}))
    return df


def _season_months(planting_month: str) -> list[str]:
    month_num = MONTH_TO_NUM.get(planting_month.title(), 1)
    months = []
    for i in range(4):
        num = ((month_num - 1 + i) % 12) + 1
        months.append(MONTHS[num - 1])
    return months


def _region_weather(region: str, survey_year: int, planting_month: str, weather_df: pd.DataFrame) -> dict[str, float]:
    season = _season_months(planting_month)
    season_rows = weather_df[(weather_df["region"] == region) & (weather_df["year"] == survey_year) & (weather_df["month"].isin(season))]
    if season_rows.empty:
        return {
            "season_mean_temp_c": np.nan,
            "season_total_rainfall_mm": np.nan,
            "season_extreme_heat_days": np.nan,
            "n_season_weather_months": 0,
        }

    return {
        "season_mean_temp_c": float(season_rows["avg_temp_c"].mean()),
        "season_total_rainfall_mm": float(season_rows["monthly_rainfall_mm"].sum()),
        "season_extreme_heat_days": float(season_rows["extreme_heat_days"].sum()),
        "n_season_weather_months": int(len(season_rows)),
    }


def _prepare_feature_row(payload: PredictionInput) -> dict[str, Any]:
    region = _standardize_label(payload.region, {"oromia": "Oromia", "oro": "Oromia", "amhara": "Amhara", "amh": "Amhara", "snnpr": "SNNPR", "snnp": "SNNPR", "tigray": "Tigray", "tig": "Tigray", "somali": "Somali", "som": "Somali"})
    crop = str(payload.crop_type).strip().lower()
    weather_df = _load_weather()
    prices_df = _load_prices()

    season_info = _region_weather(region, payload.survey_year, payload.planting_month, weather_df)
    region_temp = weather_df[weather_df["region"] == region]["avg_temp_c"].mean()
    region_rain = weather_df[weather_df["region"] == region]["monthly_rainfall_mm"].mean()

    price_row = prices_df[(prices_df["region"] == region) & (prices_df["crop_type"] == crop) & (prices_df["year"] == payload.survey_year)]
    price = float(price_row["price_birr_per_quintal"].median()) if not price_row.empty else float(prices_df[(prices_df["crop_type"] == crop) & (prices_df["year"] == payload.survey_year)]["price_birr_per_quintal"].median())

    planting_month_num = MONTH_TO_NUM.get(str(payload.planting_month).title(), 1)
    feature_row = {
        "region": region,
        "crop_type": crop,
        "planting_month": str(payload.planting_month).title(),
        "survey_year": payload.survey_year,
        "altitude_m": payload.altitude_m,
        "rainfall_mm_season": payload.rainfall_mm_season,
        "farm_size_ha": payload.farm_size_ha,
        "fertilizer_kg_per_ha": payload.fertilizer_kg_per_ha,
        "improved_seed_used": payload.improved_seed_used,
        "pest_disease_flag": payload.pest_disease_flag,
        "soil_quality_index": payload.soil_quality_index,
        "labor_days_per_ha": payload.labor_days_per_ha,
        "distance_to_market_km": payload.distance_to_market_km,
        "season_mean_temp_c": season_info["season_mean_temp_c"],
        "season_total_rainfall_mm": season_info["season_total_rainfall_mm"],
        "season_extreme_heat_days": season_info["season_extreme_heat_days"],
        "season_temp_dev_region": season_info["season_mean_temp_c"] - region_temp if not np.isnan(season_info["season_mean_temp_c"]) else np.nan,
        "season_rain_dev_region": season_info["season_total_rainfall_mm"] - region_rain if not np.isnan(season_info["season_total_rainfall_mm"]) else np.nan,
        "n_season_weather_months": season_info["n_season_weather_months"],
        "fertilizer_x_improved_seed": payload.fertilizer_kg_per_ha * payload.improved_seed_used,
        "planting_month_sin": np.sin(2 * np.pi * planting_month_num / 12),
        "planting_month_cos": np.cos(2 * np.pi * planting_month_num / 12),
        "rainfall_station_minus_plot": season_info["season_total_rainfall_mm"] - payload.rainfall_mm_season if not np.isnan(season_info["season_total_rainfall_mm"]) else np.nan,
        "price_birr_per_quintal": price,
    }
    return feature_row


def _region_crop_benchmark(region: str, crop_type: str) -> float | None:
    if not PROCESSED_TRAIN_PATH.exists():
        return None

    try:
        training = pd.read_csv(PROCESSED_TRAIN_PATH)
        subset = training[(training["region"] == region) & (training["crop_type"] == crop_type)]
        if subset.empty:
            return None
        return float(subset["yield_tons_per_ha"].mean())
    except Exception:
        return None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/data-explorer")
def data_explorer() -> dict[str, Any]:
    csv_path = ROOT / "data" / "raw" / "crop_yield_train.csv"
    if not csv_path.exists():
        return {"total_rows": 0, "regions": [], "crops": [], "sample_rows": []}

    df = pd.read_csv(csv_path)
    if df.empty:
        return {"total_rows": 0, "regions": [], "crops": [], "sample_rows": []}

    sample = df.head(12).copy()
    sample["region"] = sample["region"].str.title()
    sample["crop_type"] = sample["crop_type"].str.lower()
    sample["yield_tons_per_ha"] = sample["yield_tons_per_ha"].round(3)
    sample["rainfall_mm_season"] = sample["rainfall_mm_season"].round(1)

    summary = {
        "total_rows": int(len(df)),
        "regions": sorted(df["region"].dropna().astype(str).str.title().unique().tolist()),
        "crops": sorted(df["crop_type"].dropna().astype(str).str.lower().unique().tolist()),
        "sample_rows": sample[["region", "crop_type", "survey_year", "yield_tons_per_ha", "rainfall_mm_season"]].to_dict(orient="records"),
    }
    return summary


@app.post("/predict")
def predict(payload: PredictionInput) -> dict[str, Any]:
    feature_row = _prepare_feature_row(payload)
    df = pd.DataFrame([feature_row])
    expected = [
        "region",
        "crop_type",
        "planting_month",
        "survey_year",
        "altitude_m",
        "rainfall_mm_season",
        "farm_size_ha",
        "fertilizer_kg_per_ha",
        "improved_seed_used",
        "pest_disease_flag",
        "soil_quality_index",
        "labor_days_per_ha",
        "distance_to_market_km",
        "season_mean_temp_c",
        "season_total_rainfall_mm",
        "season_extreme_heat_days",
        "season_temp_dev_region",
        "season_rain_dev_region",
        "n_season_weather_months",
        "fertilizer_x_improved_seed",
        "planting_month_sin",
        "planting_month_cos",
        "rainfall_station_minus_plot",
    ]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    selected_model = payload.model_name.strip() if payload.model_name else "Random Forest"
    model_for_prediction, resolved_model_name = _resolve_model(selected_model)
    pred = float(model_for_prediction.predict(df[expected])[0])
    revenue = float(pred * payload.farm_size_ha * 10 * feature_row["price_birr_per_quintal"])
    benchmark = _region_crop_benchmark(feature_row["region"], feature_row["crop_type"])
    return {
        "predicted_yield_tons_per_ha": round(pred, 4),
        "estimated_revenue_birr": round(revenue, 2),
        "region": payload.region,
        "crop_type": payload.crop_type,
        "price_birr_per_quintal": round(feature_row["price_birr_per_quintal"], 2),
        "benchmark_yield_tons_per_ha": round(benchmark, 3) if benchmark is not None else None,
        "weather_summary": f"season mean temp {feature_row['season_mean_temp_c']:.1f}C, total rain {feature_row['season_total_rainfall_mm']:.0f}mm",
        "selected_model": resolved_model_name,
    }
