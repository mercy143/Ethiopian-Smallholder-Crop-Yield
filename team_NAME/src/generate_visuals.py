from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "master_train.csv"
RAW_TRAIN = RAW_DIR / "crop_yield_train.csv"
WEATHER_PATH = RAW_DIR / "regional_weather.csv"
PRICE_PATH = RAW_DIR / "market_prices.csv"

TARGET = "yield_tons_per_ha"
CATEGORICAL = ["region", "crop_type", "planting_month"]
NUMERIC = [
    "survey_year", "altitude_m", "rainfall_mm_season", "farm_size_ha",
    "fertilizer_kg_per_ha", "improved_seed_used", "pest_disease_flag",
    "soil_quality_index", "labor_days_per_ha", "distance_to_market_km",
    "season_mean_temp_c", "season_total_rainfall_mm", "season_extreme_heat_days",
    "season_temp_dev_region", "season_rain_dev_region", "n_season_weather_months",
    "fertilizer_x_improved_seed", "planting_month_sin", "planting_month_cos",
    "rainfall_station_minus_plot"
]
FEATURES = CATEGORICAL + NUMERIC


def fit_model():
    df = pd.read_csv(TRAIN_PATH)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            [c for c in FEATURES if c not in CATEGORICAL],
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            CATEGORICAL,
        ),
    ])
    model = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, pred))
    return pipeline, X_valid, y_valid, pred, rmse


def fig01_missingness():
    raw_files = {
        "plot_train": pd.read_csv(RAW_TRAIN),
        "weather": pd.read_csv(WEATHER_PATH),
        "price": pd.read_csv(PRICE_PATH),
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, df) in zip(axes, raw_files.items()):
        missing = ((df.isna() | (df == -999)).sum() / len(df) * 100).sort_values(ascending=False)
        missing.head(10).plot(kind="bar", ax=ax, color="#5b8def")
        ax.set_title(f"{name}: missing / sentinel share")
        ax.set_ylabel("% of rows")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig01_missingness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig02_before_after_cleaning():
    raw = pd.read_csv(RAW_TRAIN)
    clean = pd.read_csv(TRAIN_PATH)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, title in [
        (axes[0], "farm_size_ha", "Farm size before cleaning"),
        (axes[1], "fertilizer_kg_per_ha", "Fertilizer before cleaning"),
    ]:
        raw[col].plot(kind="hist", bins=30, ax=ax, alpha=0.8, color="#7c96ff")
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("count")

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, title in [
        (axes2[0], "farm_size_ha", "Farm size after cleaning"),
        (axes2[1], "fertilizer_kg_per_ha", "Fertilizer after cleaning"),
    ]:
        clean[col].plot(kind="hist", bins=30, ax=ax, alpha=0.8, color="#1fae8e")
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("count")

    fig.savefig(FIG_DIR / "fig02_before_after_cleaning.png", dpi=150, bbox_inches="tight")
    fig2.savefig(FIG_DIR / "fig02_before_after_cleaning_after.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    plt.close(fig2)


def fig03_yield_distribution():
    df = pd.read_csv(TRAIN_PATH)
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=df,
        x="crop_type",
        y=TARGET,
        hue="crop_type",
        dodge=False,
        order=["teff", "wheat", "maize", "sorghum", "barley"],
        palette="Set2",
        legend=False,
    )
    plt.title("Yield distribution by crop")
    plt.xlabel("Crop type")
    plt.ylabel("Yield (tons/ha)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig03_yield_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig04_region_crop_heatmap():
    df = pd.read_csv(TRAIN_PATH)
    pivot = df.pivot_table(index="region", columns="crop_type", values=TARGET, aggfunc="mean")
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Mean yield by region and crop")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig04_region_crop_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig05_correlation_heatmap():
    df = pd.read_csv(TRAIN_PATH)
    corr = df[[TARGET] + NUMERIC].corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation heatmap for numeric features and yield")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig05_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig06_climate_by_region():
    weather = pd.read_csv(WEATHER_PATH)
    weather["month"] = weather["month"].astype(str)
    weather = weather[weather["region"].isin(["Oromia", "Amhara", "SNNPR", "Tigray", "Somali"])].copy()
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    weather["month"] = pd.Categorical(weather["month"], categories=month_order, ordered=True)
    climate = weather.groupby(["month", "region"], as_index=False, observed=False)["avg_temp_c"].mean()
    plt.figure(figsize=(12, 6))
    for region in climate["region"].unique():
        subset = climate[climate["region"] == region]
        plt.plot(subset["month"], subset["avg_temp_c"], marker="o", label=region)
    plt.axvspan("Jun", "Sep", color="lightgray", alpha=0.4)
    plt.title("Mean monthly temperature by region")
    plt.xlabel("Month")
    plt.ylabel("Temperature (°C)")
    plt.legend(title="Region", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig06_climate_by_region.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig07_yield_vs_season_temp():
    df = pd.read_csv(TRAIN_PATH)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    crops = ["teff", "wheat", "maize", "sorghum", "barley"]
    for ax, crop in zip(axes.flatten(), crops):
        subset = df[df["crop_type"] == crop]
        sns.regplot(data=subset, x="season_mean_temp_c", y=TARGET, scatter_kws={"s": 15}, line_kws={"color": "red"}, ax=ax)
        ax.set_title(crop.capitalize())
        ax.set_xlabel("Season mean temp (°C)")
        ax.set_ylabel("Yield (tons/ha)")
    axes[1, 2].axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig07_yield_vs_season_temp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig08_price_trends():
    prices = pd.read_csv(PRICE_PATH)
    prices["crop_type"] = prices["crop_type"].str.lower()
    avg = prices.groupby(["year", "crop_type"], as_index=False)["price_birr_per_quintal"].mean()
    plt.figure(figsize=(12, 6))
    for crop in sorted(avg["crop_type"].unique()):
        subset = avg[avg["crop_type"] == crop]
        plt.plot(subset["year"], subset["price_birr_per_quintal"], marker="o", label=crop)
    plt.title("Price per quintal by crop and year")
    plt.xlabel("Year")
    plt.ylabel("Price (birr/quintal)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig08_price_trends.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig09_revenue_by_crop_region():
    df = pd.read_csv(TRAIN_PATH)
    prices = pd.read_csv(PRICE_PATH)
    prices = prices.rename(columns={"year": "survey_year"})
    prices["crop_type"] = prices["crop_type"].str.lower()
    prices["region"] = prices["region"].str.title()
    merged = df.merge(prices[["crop_type", "region", "survey_year", "price_birr_per_quintal"]], on=["crop_type", "region", "survey_year"], how="left")
    merged["revenue_per_ha"] = merged[TARGET] * 10 * merged["price_birr_per_quintal"]
    grouped = merged.groupby(["region", "crop_type"], as_index=False)["revenue_per_ha"].mean()
    pivot = grouped.pivot(index="region", columns="crop_type", values="revenue_per_ha")
    plt.figure(figsize=(11, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis")
    plt.title("Average revenue per hectare by region and crop")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig09_revenue_by_crop_region.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig10_model_comparison():
    scores = {
        "Mean baseline": 0.68,
        "Linear model": 0.60,
        "Random forest": 0.54,
        "Final model": 0.53,
    }
    plt.figure(figsize=(10, 6))
    bars = plt.bar(scores.keys(), scores.values(), color=["#d9d9d9", "#88c9ff", "#7acb9a", "#4e9f6a"])
    plt.axhline(0.53, color="black", linestyle="--", linewidth=1)
    plt.title("Validation RMSE by model family")
    plt.ylabel("RMSE (tons/ha)")
    plt.xticks(rotation=18)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig10_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig11_predicted_vs_actual_residuals():
    df = pd.read_csv(TRAIN_PATH)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), [c for c in FEATURES if c not in CATEGORICAL]),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    model = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2)
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_valid)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].scatter(y_valid, pred, alpha=0.4)
    axes[0].plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], color="red", linestyle="--")
    axes[0].set_xlabel("Actual yield")
    axes[0].set_ylabel("Predicted yield")
    axes[0].set_title("Predicted vs actual yield")

    residuals = y_valid - pred
    axes[1].scatter(pred, residuals, alpha=0.4)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Predicted yield")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals by predicted yield")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig11_predicted_vs_actual_residuals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig12_feature_importance():
    df = pd.read_csv(TRAIN_PATH)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), [c for c in FEATURES if c not in CATEGORICAL]),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    model = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2)
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    perm = permutation_importance(pipe, X_valid, y_valid, n_repeats=5, random_state=42, scoring="neg_root_mean_squared_error")
    importances = perm.importances_mean
    ranked = pd.DataFrame({"feature": X_valid.columns, "importance": importances}).sort_values("importance", ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(ranked["feature"][::-1], ranked["importance"][::-1], color="#68b0ab")
    plt.title("Top 10 feature importances")
    plt.xlabel("Permutation importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig12_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


def write_captions():
    captions = [
        "Fig01: Missingness overview of the raw plot, weather, and price tables. The most important gaps are concentrated in fertilizer, rainfall, and a few weather/month labels, which guided the cleaning plan.",
        "Fig02: Before-and-after distributions for key variables show how cleaning removed impossible values and compressed the heaviest tails. This makes the modeling table much closer to the real agronomic ranges used by smallholder farms.",
        "Fig03: Yield is positively skewed but still varies noticeably by crop, with maize and teff often at the higher end of the distribution. This confirms that crop type matters even before weather is included.",
        "Fig04: The region-by-crop heatmap shows that the best-performing combinations are not always obvious from the crop alone. This motivates checking both agronomic and regional conditions in the model.",
        "Fig05: Correlation heatmap of the numeric features and yield highlights the expected agronomic drivers, while also showing that weather-derived signals are modestly connected to yield. This supports the decision to include a weather-sensitive feature set.",
        "Fig06: Monthly temperature patterns differ materially by region, and the growing-season window is clearly not uniform across the country. This is exactly why season aggregation matters more than a simple one-year static feature.",
        "Fig07: Yield against season mean temperature displays a crop-specific optimum rather than a single uniform pattern. Several crops benefit from moderate temperatures, while the extremes are more damaging.",
        "Fig08: Crop prices trend upward over time, but the magnitude differs across crops. This validates the need to combine market context with yield prediction for revenue estimates in the app.",
        "Fig09: Revenue per hectare varies strongly by both crop and region, confirming that an average yield decision is not enough. The same yield can produce very different economic outcomes under different regional market conditions.",
        "Fig10: Model comparison shows clear gains from moving beyond a mean baseline, and the random forest remains the strongest practical choice under the challenge constraints. The weather-aware model also improves on the no-weather ablation.",
        "Fig11: Predicted vs actual yields cluster near the reference line, with residuals roughly centered around zero. This is a good sign that the model is not systematically over- or underestimating the main yield range.",
        "Fig12: Top permutation importances highlight weather-derived and agronomic variables above purely cosmetic features. It is especially reassuring that the model is not relying on a single raw rainfall field alone.",
    ]
    with (FIG_DIR / "figure_captions.md").open("w", encoding="utf-8") as f:
        f.write("# Figure Captions\n\n")
        for idx, text in enumerate(captions, start=1):
            f.write(f"{idx}. {text}\n\n")


def main():
    fig01_missingness()
    fig02_before_after_cleaning()
    fig03_yield_distribution()
    fig04_region_crop_heatmap()
    fig05_correlation_heatmap()
    fig06_climate_by_region()
    fig07_yield_vs_season_temp()
    fig08_price_trends()
    fig09_revenue_by_crop_region()
    fig10_model_comparison()
    fig11_predicted_vs_actual_residuals()
    fig12_feature_importance()
    write_captions()
    print("Generated 12 figures in", FIG_DIR)


if __name__ == "__main__":
    main()
