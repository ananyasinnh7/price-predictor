"""Model and explainability helpers for price prediction."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


FEATURE_COLUMNS = ["size_sqft", "bedrooms", "bathrooms", "age_years", "distance_to_city_km"]
TARGET_COLUMN = "price"


def get_sample_data() -> pd.DataFrame:
    """Return a compact sample real-estate dataset."""
    data = [
        [900, 2, 1, 25, 12, 180000],
        [1100, 3, 2, 15, 9, 240000],
        [1400, 3, 2, 8, 7, 310000],
        [1600, 4, 2, 12, 6, 330000],
        [2000, 4, 3, 5, 5, 420000],
        [2300, 5, 3, 3, 4, 495000],
        [1250, 3, 2, 20, 11, 255000],
        [1750, 4, 2, 10, 8, 355000],
        [2100, 4, 3, 7, 6, 430000],
        [2600, 5, 4, 2, 3, 560000],
    ]
    return pd.DataFrame(data, columns=[*FEATURE_COLUMNS, TARGET_COLUMN])


def train_model(dataframe: pd.DataFrame) -> RandomForestRegressor:
    """Train a regression model on the provided dataframe."""
    x = dataframe[FEATURE_COLUMNS]
    y = dataframe[TARGET_COLUMN]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(x, y)
    return model


def predict_price(model: RandomForestRegressor, features: pd.DataFrame) -> float:
    """Predict price for a single-row feature dataframe."""
    prediction = model.predict(features[FEATURE_COLUMNS])[0]
    return float(prediction)


def feature_importance_for_input(
    model: RandomForestRegressor, features: pd.DataFrame
) -> pd.DataFrame:
    """Return explainability values for charting.

    Uses SHAP if installed; otherwise falls back to model feature importances.
    """
    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features[FEATURE_COLUMNS])
        values = shap_values[0]
        importances = [abs(float(v)) for v in values]
    except Exception:
        importances = [float(v) for v in model.feature_importances_]

    return (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

