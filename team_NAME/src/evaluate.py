"""Evaluate the crop-yield model and compare a weather-aware model with a no-weather ablation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from cleaning import build_master_tables, clean_plot_table, clean_price_table, clean_weather_table
from train import FEATURES, TARGET, build_model_pipeline, load_raw_data

ROOT = Path(__file__).resolve().parents[1]

WEATHER_FEATURES = [
    "season_mean_temp_c",
    "season_total_rainfall_mm",
    "season_extreme_heat_days",
    "season_temp_dev_region",
    "season_rain_dev_region",
    "n_season_weather_months",
    "planting_month_sin",
    "planting_month_cos",
    "rainfall_station_minus_plot",
]
NON_WEATHER_FEATURES = [feature for feature in FEATURES if feature not in WEATHER_FEATURES]


def evaluate_model(feature_set: list[str], label: str) -> tuple[float, float, float]:
    train_raw, _, weather_raw, price_raw = load_raw_data()
    train_clean = clean_plot_table(train_raw)
    weather_clean = clean_weather_table(weather_raw)
    price_clean = clean_price_table(price_raw)

    empty_test = train_clean.iloc[:0].copy()
    if "plot_id" not in empty_test.columns:
        empty_test["plot_id"] = pd.Series(dtype="object")

    train_master, _ = build_master_tables(train_clean, empty_test, weather_clean, price_clean)

    train_master = train_master[feature_set + [TARGET]].dropna(subset=feature_set + [TARGET]).copy()
    X = train_master[feature_set]
    y = train_master[TARGET]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_model_pipeline(feature_set)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_valid)

    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    mae = mean_absolute_error(y_valid, preds)
    r2 = r2_score(y_valid, preds)
    print(f"{label}: RMSE={rmse:.4f}, MAE={mae:.4f}, R^2={r2:.4f}")
    return rmse, mae, r2


def main() -> None:
    print("Evaluating weather-aware vs no-weather ablation...")
    evaluate_model(FEATURES, "Weather-aware model")
    evaluate_model(NON_WEATHER_FEATURES, "No-weather ablation")


if __name__ == "__main__":
    main()
