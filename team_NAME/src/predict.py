"""Generate a submission file using the trained regression model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from cleaning import clean_plot_table, clean_price_table, clean_weather_table, build_master_tables
from train import FEATURES, TARGET, load_raw_data

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "models" / "final_model.joblib"
SUBMISSION_PATH = ROOT / "submission" / "team_NAME_submission.csv"


def main() -> None:
    train_raw, test_raw, weather_raw, price_raw = load_raw_data()

    train_clean = clean_plot_table(train_raw)
    test_clean = clean_plot_table(test_raw)
    weather_clean = clean_weather_table(weather_raw)
    price_clean = clean_price_table(price_raw)
    _, test_master = build_master_tables(train_clean, test_clean, weather_clean, price_clean)

    test_master = test_master[FEATURES].copy()
    model = joblib.load(MODEL_PATH)
    predictions = model.predict(test_master)

    submission = pd.DataFrame(
        {
            "plot_id": test_raw["plot_id"],
            "predicted_yield_tons_per_ha": predictions,
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved {len(submission)} predictions to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
