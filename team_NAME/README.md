# Ethiopian Smallholder Crop Yield Challenge

This project contains the cleaned data pipeline, feature engineering, regression model training, and final submission generation for the Ethiopian Smallholder Crop-Yield hackathon.

## Team / project

- Team folder: team_NAME
- Objective: predict yield_tons_per_ha using plot-level information, weather signals, and price context while avoiding leakage and fitting all preprocessing on the training data only.

## Project structure

- data/raw/ — original CSV files, never edited
- data/processed/ — processed master_train.csv, master_test.csv, and the data dictionary
- notebooks/ — A–D notebooks for cleaning, analysis, figures, and evaluation
- src/ — reusable Python scripts for cleaning, training, and prediction
- models/ — trained model artifact
- figures/ — PNG output and figure captions
- reports/ — exported analysis reports
- app/ — demo app entry point and bundled assets
- submission/ — final prediction file
- presentation/ — slides and presentation materials

## Setup

From the project root:

```bash
python -m venv .venv
. .venv/bin/activate   # Linux/macOS
# or .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Run order

```bash
python src/train.py
python src/predict.py
```

## Local app run

Start the backend and frontend in separate terminals:

```powershell
cd "c:\Users\Administrator\Desktop\Ethiopian Smallholder Crop-Yield\team_NAME\app\backend"
C:/Users/Administrator/AppData/Local/Programs/Python/Python312/python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

```powershell
cd "c:\Users\Administrator\Desktop\Ethiopian Smallholder Crop-Yield\team_NAME\app\frontend"
npm run dev -- --hostname 0.0.0.0 --port 3000
```

Open http://localhost:3000

## What each script does

- src/train.py: cleans the raw train/test tables, builds engineered features, trains the regression model, and writes processed CSVs and the model artifact.
- src/predict.py: loads the trained model, prepares the leaderboard test set with the same cleaning/feature pipeline, and writes the submission file.

## Outputs

- models/final_model.joblib
- data/processed/master_train.csv
- data/processed/master_test.csv
- submission/team_NAME_submission.csv

## Validation status

The training script was run successfully with fresh output:

- Validation RMSE: 0.5244
- Validation MAE: 0.3855
- Validation R²: 0.8623
