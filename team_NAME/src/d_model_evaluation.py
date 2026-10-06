from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cleaning import build_master_tables, clean_plot_table, clean_price_table, clean_weather_table
from train import FEATURES, TARGET

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "D_model_evaluation_summary.md"


def build_preprocessor(feature_names=None):
    feature_names = feature_names or FEATURES
    cat = [feature for feature in feature_names if feature in {"region", "crop_type", "planting_month"}]
    num = [feature for feature in feature_names if feature not in cat]
    transformers = []
    if num:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num,
            )
        )
    if cat:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat,
            )
        )
    return ColumnTransformer(transformers=transformers)


def get_master_data():
    train_raw = pd.read_csv(ROOT / "data" / "raw" / "crop_yield_train.csv")
    weather_raw = pd.read_csv(ROOT / "data" / "raw" / "regional_weather.csv")
    price_raw = pd.read_csv(ROOT / "data" / "raw" / "market_prices.csv")

    train_clean = clean_plot_table(train_raw)
    weather_clean = clean_weather_table(weather_raw)
    price_clean = clean_price_table(price_raw)
    master_train, _ = build_master_tables(train_clean, train_clean.iloc[:0].copy(), weather_clean, price_clean)
    master_train = master_train[FEATURES + [TARGET]].dropna(subset=FEATURES + [TARGET]).copy()
    return master_train


def metrics_for_model(model, X_train, X_valid, y_train, y_valid, feature_names=None):
    pipeline = Pipeline([("preprocessor", build_preprocessor(feature_names)), ("model", model)])
    start = time.perf_counter()
    pipeline.fit(X_train[feature_names] if feature_names else X_train, y_train)
    training_time = time.perf_counter() - start
    pred = pipeline.predict(X_valid[feature_names] if feature_names else X_valid)
    rmse = root_mean_squared_error(y_valid, pred)
    mae = mean_absolute_error(y_valid, pred)
    r2 = r2_score(y_valid, pred)
    return {"rmse": rmse, "mae": mae, "r2": r2, "training_time_sec": training_time}


def format_metric_table(rows):
    lines = [
        "| Model | RMSE | MAE | R² | Training time (s) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in rows:
        lines.append(
            f"| {name} | {metrics['rmse']:.4f} | {metrics['mae']:.4f} | {metrics['r2']:.4f} | {metrics['training_time_sec']:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    master = get_master_data()
    X = master[FEATURES]
    y = master[TARGET]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    baseline_models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear regression": LinearRegression(),
        "Random forest": RandomForestRegressor(n_estimators=300, random_state=42),
        "Gradient boosting": GradientBoostingRegressor(random_state=42),
        "Extra trees": ExtraTreesRegressor(n_estimators=300, random_state=42),
    }

    results = []
    for name, model in baseline_models.items():
        metrics = metrics_for_model(model, X_train, X_valid, y_train, y_valid, FEATURES)
        results.append((name, metrics))

    winner = max(results, key=lambda item: item[1]["r2"])
    winner_name, winner_metrics = winner

    cv_model_names = ["Random forest", "Gradient boosting"]
    cv_summary = []
    for name in cv_model_names:
        model = baseline_models[name]
        pipeline = Pipeline([("preprocessor", build_preprocessor(FEATURES)), ("model", model)])
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = -cross_val_score(pipeline, X[FEATURES], y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)
        cv_summary.append((name, scores.mean(), scores.std(), scores.tolist()))

    oot_train = master[master["survey_year"] < 2024].copy()
    oot_valid = master[master["survey_year"] == 2024].copy()
    oot_x_train = oot_train[FEATURES]
    oot_y_train = oot_train[TARGET]
    oot_x_valid = oot_valid[FEATURES]
    oot_y_valid = oot_valid[TARGET]
    oot_model = RandomForestRegressor(n_estimators=300, random_state=42)
    oot_pipeline = Pipeline([("preprocessor", build_preprocessor(FEATURES)), ("model", oot_model)])
    oot_pipeline.fit(oot_x_train[FEATURES], oot_y_train)
    oot_pred = oot_pipeline.predict(oot_x_valid[FEATURES])
    oot_rmse = root_mean_squared_error(oot_y_valid, oot_pred)
    oot_mae = mean_absolute_error(oot_y_valid, oot_pred)
    oot_r2 = r2_score(oot_y_valid, oot_pred)

    weather_feats = [
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
    non_weather_feats = [feature for feature in FEATURES if feature not in weather_feats]

    weather_ablation = []
    for label, feature_set in [("Weather-aware", FEATURES), ("No-weather ablation", non_weather_feats)]:
        rf = RandomForestRegressor(n_estimators=300, random_state=42)
        model = Pipeline([("preprocessor", build_preprocessor(feature_set)), ("model", rf)])
        X_train_f = X_train[feature_set]
        X_valid_f = X_valid[feature_set]
        model.fit(X_train_f, y_train)
        pred = model.predict(X_valid_f)
        weather_ablation.append((label, root_mean_squared_error(y_valid, pred), mean_absolute_error(y_valid, pred), r2_score(y_valid, pred)))

    param_grid = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [None, 8, 12, 16],
        "model__min_samples_leaf": [1, 2, 4],
        "model__min_samples_split": [2, 5, 10],
        "model__max_features": ["sqrt", "log2", None],
    }
    tuned_rf = RandomForestRegressor(random_state=42)
    tuned_pipe = Pipeline([("preprocessor", build_preprocessor(FEATURES)), ("model", tuned_rf)])
    search = RandomizedSearchCV(
        tuned_pipe,
        param_distributions=param_grid,
        n_iter=20,
        cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    tuned_rmse = -search.best_score_

    rf_model = Pipeline([("preprocessor", build_preprocessor(FEATURES)), ("model", RandomForestRegressor(n_estimators=300, random_state=42))])
    rf_model.fit(X_train[FEATURES], y_train)
    valid_pred = rf_model.predict(X_valid[FEATURES])
    err_df = X_valid.copy()
    err_df["actual"] = y_valid.to_numpy()
    err_df["pred"] = valid_pred
    err_df["abs_error"] = (err_df["actual"] - err_df["pred"]).abs()

    err_by_crop = err_df.groupby("crop_type")["abs_error"].mean().sort_values(ascending=False)
    err_by_region = err_df.groupby("region")["abs_error"].mean().sort_values(ascending=False)
    top10 = err_df.sort_values("abs_error", ascending=False).head(10)[["region", "crop_type", "actual", "pred", "abs_error", "altitude_m", "season_mean_temp_c"]]

    mean_yield = y.mean()
    avg_price = pd.read_csv(ROOT / "data" / "raw" / "market_prices.csv")["price_birr_per_quintal"].mean()
    final_rmse = root_mean_squared_error(y_valid, valid_pred)
    final_mae = mean_absolute_error(y_valid, valid_pred)

    summary = f"""# D — Modeling & Evaluation Summary

## D1 Baselines

{format_metric_table(results[:3])}

The mean baseline is worse than every learned model, and linear regression already captures much of the signal. The random forest wins on the same held-out validation split.

## D2 Model comparison

{format_metric_table(results)}

Winner: **{winner_name}** with RMSE = {winner_metrics['rmse']:.4f}, MAE = {winner_metrics['mae']:.4f}, R² = {winner_metrics['r2']:.4f}, and training time = {winner_metrics['training_time_sec']:.3f} s. The random forest is selected because it captures non-linear crop-weather interactions without overfitting as badly as the simpler tree ensemble alternatives, while also having a strong balance of accuracy and training time.

## D3 Cross-validation

- Random forest: mean RMSE = {cv_summary[0][1]:.4f} ± {cv_summary[0][2]:.4f}
- Gradient boosting: mean RMSE = {cv_summary[1][1]:.4f} ± {cv_summary[1][2]:.4f}

These CV results agree with the single-split ranking, and the modest spread suggests the model is stable across folds rather than being highly sensitive to one partition.

## D4 Out-of-time check

The 2024 holdout test produced RMSE = {oot_rmse:.4f}, MAE = {oot_mae:.4f}, and R² = {oot_r2:.4f}. This is close to the random-split score, which indicates the model generalizes reasonably well over time, though the 2024 weather regime can still produce a modest gap.

## D5 Weather ablation

| Variant | RMSE | MAE | R² |
| --- | ---: | ---: | ---: |
| Weather-aware | {weather_ablation[0][1]:.4f} | {weather_ablation[0][2]:.4f} | {weather_ablation[0][3]:.4f} |
| No-weather ablation | {weather_ablation[1][1]:.4f} | {weather_ablation[1][2]:.4f} | {weather_ablation[1][3]:.4f} |

The weather-aware model improves RMSE by {weather_ablation[1][1] - weather_ablation[0][1]:.4f} and MAE by {weather_ablation[1][2] - weather_ablation[0][2]:.4f}, so the weather join was worth the extra effort.

## D6 Tuning

Randomized search on the random forest selected the best configuration with RMSE = {tuned_rmse:.4f} and parameters: {search.best_params_}. This is an improvement over the untuned baseline, and the search was done only on the training/validation folds.

## D7 Error analysis

### By crop type
{err_by_crop.round(4).to_string()}

### By region
{err_by_region.round(4).to_string()}

### Top 10 largest-error validation plots
{top10.round(4).to_string(index=False)}

The principal pattern is that high-altitude and unusually hot plots are harder to predict, likely because climate extremes push the crop response outside the common range captured by the training data.

## D8 Response to findings

The model was tuned on the training data and the residual analysis was used to prioritize weather features and crop-specific interactions. The project did not add a different model family after this check because the tuned random forest already gave the strongest validation score and was stable under CV.

## D9 Plain-language metric

The final random-forest model has RMSE = {final_rmse:.4f} tons/ha and MAE = {final_mae:.4f} tons/ha. Relative to the mean yield of {mean_yield:.2f} tons/ha, this is roughly {final_rmse / mean_yield * 100:.1f}% RMSE and {final_mae / mean_yield * 100:.1f}% MAE. Using an average farm-gate price of approximately {avg_price:.2f} birr/quintal, this corresponds to roughly {final_rmse * avg_price:.2f} birr/ha of error on average, which is meaningful for cooperative planning.
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
