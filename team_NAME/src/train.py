"""Train a regression model for the Ethiopian crop-yield challenge."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cleaning import clean_plot_table, clean_price_table, clean_weather_table, build_master_tables

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
SUBMISSION_DIR = ROOT / "submission"

TARGET = "yield_tons_per_ha"
CATEGORICAL_FEATURES = ["region", "crop_type", "planting_month"]
NUMERICAL_FEATURES = [
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
FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(RAW_DIR / "crop_yield_train.csv")
    test = pd.read_csv(RAW_DIR / "crop_yield_leaderboard_test.csv")
    weather = pd.read_csv(RAW_DIR / "regional_weather.csv")
    prices = pd.read_csv(RAW_DIR / "market_prices.csv")
    return train, test, weather, prices


def build_model_pipeline(feature_names: list[str] | None = None, model=None) -> Pipeline:
    feature_names = FEATURES if feature_names is None else feature_names
    numerical_features = [feature for feature in feature_names if feature not in CATEGORICAL_FEATURES]
    categorical_features = [feature for feature in feature_names if feature in CATEGORICAL_FEATURES]

    transformers = [
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numerical_features,
        ),
    ]
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    if model is None:
        model = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=42)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def main() -> None:
    train_raw, test_raw, weather_raw, price_raw = load_raw_data()

    train_clean = clean_plot_table(train_raw)
    test_clean = clean_plot_table(test_raw)
    weather_clean = clean_weather_table(weather_raw)
    price_clean = clean_price_table(price_raw)

    train_master, test_master = build_master_tables(train_clean, test_clean, weather_clean, price_clean)

    train_master = train_master[FEATURES + [TARGET]].copy()
    test_master = test_master[FEATURES].copy()

    X = train_master[FEATURES]
    y = train_master[TARGET]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline = build_model_pipeline()
    pipeline.fit(X_train, y_train)
    valid_pred = pipeline.predict(X_valid)

    rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
    mae = mean_absolute_error(y_valid, valid_pred)
    r2 = r2_score(y_valid, valid_pred)

    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation R^2: {r2:.4f}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_master.to_csv(PROCESSED_DIR / "master_train.csv", index=False)
    test_master.to_csv(PROCESSED_DIR / "master_test.csv", index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_variants = {
        "random_forest": RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "extra_trees": ExtraTreesRegressor(n_estimators=300, random_state=42),
        "linear_regression": LinearRegression(),
    }
    for filename, estimator in model_variants.items():
        variant_pipeline = build_model_pipeline(model=estimator)
        variant_pipeline.fit(X_train, y_train)
        joblib.dump(variant_pipeline, MODEL_DIR / f"{filename}.joblib")
    joblib.dump(pipeline, MODEL_DIR / "final_model.joblib")

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    preds = pipeline.predict(test_master)
    submission = pd.DataFrame({"plot_id": test_raw["plot_id"], "predicted_yield_tons_per_ha": preds})
    submission.to_csv(SUBMISSION_DIR / "team_NAME_submission.csv", index=False)


if __name__ == "__main__":
    main()
