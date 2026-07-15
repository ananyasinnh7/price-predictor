# price-predictor

An explainable real estate price predictor with a lightweight web dashboard.

## Features
- Trains a regression model for house prices.
- Predicts a price from user-entered features.
- Shows **why** via a feature-importance chart.
- Uses SHAP values when available, with a safe fallback to model importances.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run dashboard
```bash
streamlit run /home/runner/work/price-predictor/price-predictor/app.py
```

## Run tests
```bash
python -m unittest discover -s /home/runner/work/price-predictor/price-predictor/tests
```