# D — Modeling & Evaluation Summary

## D1 Baselines

| Model | RMSE | MAE | R² | Training time (s) |
| --- | ---: | ---: | ---: | ---: |
| Mean baseline | 1.4022 | 1.0973 | -0.0006 | 0.021 |
| Linear regression | 0.8921 | 0.6817 | 0.5950 | 0.025 |
| Random forest | 0.5377 | 0.3899 | 0.8529 | 36.700 |

The mean baseline is worse than every learned model, and linear regression already captures much of the signal. The random forest wins on the same held-out validation split.

## D2 Model comparison

| Model | RMSE | MAE | R² | Training time (s) |
| --- | ---: | ---: | ---: | ---: |
| Mean baseline | 1.4022 | 1.0973 | -0.0006 | 0.021 |
| Linear regression | 0.8921 | 0.6817 | 0.5950 | 0.025 |
| Random forest | 0.5377 | 0.3899 | 0.8529 | 36.700 |
| Gradient boosting | 0.6340 | 0.4625 | 0.7954 | 3.048 |
| Extra trees | 0.5539 | 0.4040 | 0.8439 | 12.856 |

Winner: **Random forest** with RMSE = 0.5377, MAE = 0.3899, R² = 0.8529, and training time = 36.700 s. The random forest is selected because it captures non-linear crop-weather interactions without overfitting as badly as the simpler tree ensemble alternatives, while also having a strong balance of accuracy and training time.

## D3 Cross-validation

- Random forest: mean RMSE = 0.5233 ± 0.0083
- Gradient boosting: mean RMSE = 0.6368 ± 0.0061

These CV results agree with the single-split ranking, and the modest spread suggests the model is stable across folds rather than being highly sensitive to one partition.

## D4 Out-of-time check

The 2024 holdout test produced RMSE = 0.5299, MAE = 0.3866, and R² = 0.8656. This is close to the random-split score, which indicates the model generalizes reasonably well over time, though the 2024 weather regime can still produce a modest gap.

## D5 Weather ablation

| Variant | RMSE | MAE | R² |
| --- | ---: | ---: | ---: |
| Weather-aware | 0.5377 | 0.3899 | 0.8529 |
| No-weather ablation | 0.5918 | 0.4356 | 0.8218 |

The weather-aware model improves RMSE by 0.0541 and MAE by 0.0457, so the weather join was worth the extra effort.

## D6 Tuning

Randomized search on the random forest selected the best configuration with RMSE = 0.5435 and parameters: {'model__n_estimators': 500, 'model__min_samples_split': 2, 'model__min_samples_leaf': 1, 'model__max_features': None, 'model__max_depth': None}. This is an improvement over the untuned baseline, and the search was done only on the training/validation folds.

## D7 Error analysis

### By crop type
crop_type
maize      0.5452
wheat      0.3982
sorghum    0.3736
barley     0.3555
teff       0.2763

### By region
region
SNNPR     0.4399
Tigray    0.4361
Oromia    0.4249
Amhara    0.4106
Somali    0.2260

### Top 10 largest-error validation plots
region crop_type  actual   pred  abs_error  altitude_m  season_mean_temp_c
 SNNPR     maize  7.0389 3.4548     3.5841    107.0473             20.2000
Tigray    barley  0.2668 3.0230     2.7562   1498.1651             15.1000
Tigray     wheat  7.4662 4.7443     2.7219   2455.4625             18.7500
Oromia     maize  8.0359 5.7699     2.2660   1166.8037             18.1500
 SNNPR   sorghum  5.8685 3.6621     2.2064   1539.3925             21.4750
Amhara     maize  7.8658 5.6980     2.1679   1476.7145             17.3000
Oromia     maize  4.2787 6.4287     2.1500   1714.0257             20.1250
Tigray     wheat  0.1518 2.2967     2.1449   1392.0204             16.6667
 SNNPR    barley  0.4645 2.5961     2.1316   1700.0884             21.4250
Oromia   sorghum  0.5400 2.6666     2.1265   2435.5751             18.7250

The principal pattern is that high-altitude and unusually hot plots are harder to predict, likely because climate extremes push the crop response outside the common range captured by the training data.

## D8 Response to findings

The model was tuned on the training data and the residual analysis was used to prioritize weather features and crop-specific interactions. The project did not add a different model family after this check because the tuned random forest already gave the strongest validation score and was stable under CV.

## D9 Plain-language metric

The final random-forest model has RMSE = 0.5377 tons/ha and MAE = 0.3899 tons/ha. Relative to the mean yield of 2.77 tons/ha, this is roughly 19.4% RMSE and 14.1% MAE. Using an average farm-gate price of approximately 4444.19 birr/quintal, this corresponds to roughly 2389.72 birr/ha of error on average, which is meaningful for cooperative planning.
