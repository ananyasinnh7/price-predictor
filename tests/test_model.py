import unittest

import pandas as pd

from price_predictor.model import (
    FEATURE_COLUMNS,
    feature_importance_for_input,
    get_sample_data,
    predict_price,
    train_model,
)


class ModelTests(unittest.TestCase):
    def test_predict_price_returns_float(self):
        dataset = get_sample_data()
        model = train_model(dataset)
        features = dataset[FEATURE_COLUMNS].iloc[[0]]
        prediction = predict_price(model, features)
        self.assertIsInstance(prediction, float)
        self.assertGreater(prediction, 0.0)

    def test_feature_importance_shape_and_columns(self):
        dataset = get_sample_data()
        model = train_model(dataset)
        features = pd.DataFrame(
            [[1500, 3, 2, 10, 8]],
            columns=FEATURE_COLUMNS,
        )
        importance_df = feature_importance_for_input(model, features)
        self.assertEqual(list(importance_df.columns), ["feature", "importance"])
        self.assertEqual(len(importance_df), len(FEATURE_COLUMNS))
        self.assertTrue((importance_df["importance"] >= 0).all())


if __name__ == "__main__":
    unittest.main()

