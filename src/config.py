"""Shared constants and paths. Relative paths only."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
SUBMISSION = ROOT / "submission"
APP_ASSETS = ROOT / "app" / "assets"
NOTEBOOKS = ROOT / "notebooks"
PRESENTATION = ROOT / "presentation"

TEAM_NAME = "team_addis"
RANDOM_STATE = 42

ALLOWED_REGIONS = ["Amhara", "Oromia", "SNNPR", "Somali", "Tigray"]
ALLOWED_CROPS = ["barley", "maize", "sorghum", "teff", "wheat"]

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

MONTH_TO_NUM = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
NUM_TO_MONTH = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

# Planting month plus the following three months (same calendar year for this survey).
GROWING_SEASON_MONTHS = 4

CATEGORICAL_FEATURES = ["region", "crop_type", "planting_month"]
NUMERIC_FEATURES = [
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
WEATHER_FEATURES = [
    "season_mean_temp_c",
    "season_total_rainfall_mm",
    "season_extreme_heat_days",
    "season_temp_dev_region",
    "season_rain_dev_region",
    "n_season_weather_months",
    "rainfall_station_minus_plot",
]
TARGET = "yield_tons_per_ha"

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
