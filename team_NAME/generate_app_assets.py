from pathlib import Path

import pandas as pd

base = Path(__file__).resolve().parent
weather = pd.read_csv(base / "data" / "raw" / "regional_weather.csv")
price = pd.read_csv(base / "data" / "raw" / "market_prices.csv")

region_alias = {
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

crop_alias = {
    "teff": "teff",
    "tef": "teff",
    "wheat": "wheat",
    "maize": "maize",
    "corn": "maize",
    "sorghum": "sorghum",
    "barley": "barley",
}

month_alias = {
    "jan": "Jan",
    "january": "Jan",
    "feb": "Feb",
    "february": "Feb",
    "mar": "Mar",
    "march": "Mar",
    "apr": "Apr",
    "april": "Apr",
    "may": "May",
    "jun": "Jun",
    "june": "Jun",
    "jul": "Jul",
    "july": "Jul",
    "aug": "Aug",
    "august": "Aug",
    "sep": "Sep",
    "sept": "Sep",
    "september": "Sep",
    "oct": "Oct",
    "october": "Oct",
    "nov": "Nov",
    "november": "Nov",
    "dec": "Dec",
    "december": "Dec",
}

weather["region"] = weather["region"].map(lambda x: region_alias.get(str(x).strip().lower(), str(x).strip().title()))
weather["month"] = weather["month"].map(lambda x: month_alias.get(str(x).strip().lower(), str(x).strip().title()))
weather = weather.drop_duplicates().copy()
weather["avg_temp_c"] = pd.to_numeric(weather["avg_temp_c"], errors="coerce")
weather["monthly_rainfall_mm"] = pd.to_numeric(weather["monthly_rainfall_mm"], errors="coerce")
weather["extreme_heat_days"] = pd.to_numeric(weather["extreme_heat_days"], errors="coerce").fillna(0)

price["crop_type"] = price["crop_type"].map(lambda x: crop_alias.get(str(x).strip().lower(), str(x).strip().lower()))
price["region"] = price["region"].map(lambda x: region_alias.get(str(x).strip().lower(), str(x).strip().title()))
price["price_birr_per_quintal"] = pd.to_numeric(price["price_birr_per_quintal"], errors="coerce")
price["price_birr_per_quintal"] = price.groupby(["crop_type", "year"])["price_birr_per_quintal"].transform(lambda s: s.fillna(s.median()))
price = price.drop_duplicates(subset=["crop_type", "region", "year"], keep="first")

asset_dir = base / "app" / "assets"
asset_dir.mkdir(parents=True, exist_ok=True)
weather.to_csv(asset_dir / "weather_clean.csv", index=False)
price.to_csv(asset_dir / "prices_clean.csv", index=False)

print(f"weather rows: {len(weather)}")
print(f"price rows: {len(price)}")
