# Figure Captions

1. Fig01: Missingness overview of the raw plot, weather, and price tables. The most important gaps are concentrated in fertilizer, rainfall, and a few weather/month labels, which guided the cleaning plan.

2. Fig02: Before-and-after distributions for key variables show how cleaning removed impossible values and compressed the heaviest tails. This makes the modeling table much closer to the real agronomic ranges used by smallholder farms.

3. Fig03: Yield is positively skewed but still varies noticeably by crop, with maize and teff often at the higher end of the distribution. This confirms that crop type matters even before weather is included.

4. Fig04: The region-by-crop heatmap shows that the best-performing combinations are not always obvious from the crop alone. This motivates checking both agronomic and regional conditions in the model.

5. Fig05: Correlation heatmap of the numeric features and yield highlights the expected agronomic drivers, while also showing that weather-derived signals are modestly connected to yield. This supports the decision to include a weather-sensitive feature set.

6. Fig06: Monthly temperature patterns differ materially by region, and the growing-season window is clearly not uniform across the country. This is exactly why season aggregation matters more than a simple one-year static feature.

7. Fig07: Yield against season mean temperature displays a crop-specific optimum rather than a single uniform pattern. Several crops benefit from moderate temperatures, while the extremes are more damaging.

8. Fig08: Crop prices trend upward over time, but the magnitude differs across crops. This validates the need to combine market context with yield prediction for revenue estimates in the app.

9. Fig09: Revenue per hectare varies strongly by both crop and region, confirming that an average yield decision is not enough. The same yield can produce very different economic outcomes under different regional market conditions.

10. Fig10: Model comparison shows clear gains from moving beyond a mean baseline, and the random forest remains the strongest practical choice under the challenge constraints. The weather-aware model also improves on the no-weather ablation.

11. Fig11: Predicted vs actual yields cluster near the reference line, with residuals roughly centered around zero. This is a good sign that the model is not systematically over- or underestimating the main yield range.

12. Fig12: Top permutation importances highlight weather-derived and agronomic variables above purely cosmetic features. It is especially reassuring that the model is not relying on a single raw rainfall field alone.

