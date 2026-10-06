from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
OUT = ROOT / "presentation" / "team_NAME_slides.pptx"
OUT.parent.mkdir(parents=True, exist_ok=True)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Slide 1
slide = prs.slides.add_slide(prs.slide_layouts[0])
slide.shapes.title.text = "Ethiopian Smallholder Crop Yield Challenge"
subtitle = slide.shapes.placeholders[1]
subtitle.text = "Weather-aware yield prediction, pricing context, and live demo"

# Slide 2
slide = prs.slides.add_slide(prs.slide_layouts[1])
slide.shapes.title.text = "Data cleaning and integration"
body = slide.shapes.placeholders[1].text_frame
body.text = (
    "• Cleaned plot, weather, and market tables in code\n"
    "• Standardized region and crop labels across all tables\n"
    "• Joined weather by region + year + planting-season window\n"
    "• Added weather-derived features and revenue metrics\n"
    "• Validated model features on train-only statistics"
)

# Slide 3
slide = prs.slides.add_slide(prs.slide_layouts[5])
slide.shapes.title.text = "Key findings"
left = slide.shapes.add_textbox(Inches(0.5), Inches(1.3), Inches(5.5), Inches(3.5))
left.text_frame.text = (
    "• Yield differs materially by crop and region\n"
    "• Weather features improve the model versus the no-weather baseline\n"
    "• Price trends show material regional and crop differences\n"
    "• Revenue per hectare is not the same as yield per hectare"
)
img = slide.shapes.add_picture(str(FIG_DIR / "fig04_region_crop_heatmap.png"), Inches(6.2), Inches(1.5), Inches(6.2), Inches(4.4))

# Slide 4
slide = prs.slides.add_slide(prs.slide_layouts[5])
slide.shapes.title.text = "Modeling and evaluation"
left = slide.shapes.add_textbox(Inches(0.5), Inches(1.3), Inches(5.2), Inches(3.2))
left.text_frame.text = (
    "Weather-aware model: RMSE 0.5397, MAE 0.3908, R² 0.8518\n"
    "No-weather ablation: RMSE 0.5933, MAE 0.4352, R² 0.8209\n"
    "Final choice: RandomForestRegressor with weather features\n"
    "Train/validation split: 80/20, random_state=42"
)
img = slide.shapes.add_picture(str(FIG_DIR / "fig10_model_comparison.png"), Inches(6.2), Inches(1.4), Inches(6.2), Inches(4.3))

# Slide 5
slide = prs.slides.add_slide(prs.slide_layouts[1])
slide.shapes.title.text = "Demo and next steps"
body = slide.shapes.placeholders[1].text_frame
body.text = (
    "• Live prediction app served on localhost:3000 and API on localhost:8000\n"
    "• The app looks up seasonal weather and price automatically\n"
    "• Estimated yield and revenue are returned in one step\n"
    "• Next improvement: expand tuning and uncertainty estimates\n"
    "• Link: local demo run from the project root and app directory"
)

prs.save(OUT)
print(f"Saved presentation to {OUT}")
