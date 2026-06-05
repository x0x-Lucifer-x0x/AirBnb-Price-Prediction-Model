"""
Inference
=========
Load saved artifacts and run predictions on new/unseen listings.

Usage (Python):
    from src.inference import PriceInferenceEngine
    engine = PriceInferenceEngine("outputs/")
    predictions = engine.predict(new_listings_df)

Usage (CLI):
    python -m src.inference --input new_listings.csv --output predictions.csv
"""

import argparse
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import load_artifact, setup_logging
from src.feature_engineering import FeatureEngineer
from src.data_loader import DataLoader

logger = logging.getLogger(__name__)


class PriceInferenceEngine:
    def __init__(self, output_dir: str = "outputs/", data_dir: str = "data/"):
        self.output_dir = Path(output_dir)
        self.data_dir   = Path(data_dir)

        logger.info("Loading saved pipeline artifacts …")
        self.preprocessor = load_artifact(output_dir, "preprocessor.pkl")
        self.predictor     = load_artifact(output_dir, "model.pkl")

        # Need geo data for FeatureEngineer
        loader = DataLoader(data_dir)
        _, _, _, geo_df = loader.load_all()
        self.engineer = FeatureEngineer(geo_df)
        # Reuse fitted KMeans from training
        saved_engineer = load_artifact(output_dir, "feature_engineer.pkl") \
            if (self.output_dir / "feature_engineer.pkl").exists() else None
        if saved_engineer:
            self.engineer._geo_kmeans  = saved_engineer._geo_kmeans
            self.engineer._neigh_stats = saved_engineer._neigh_stats

    # ------------------------------------------------------------------ #
    def predict(
        self,
        listings_df: pd.DataFrame,
        reviews_df: pd.DataFrame | None = None,
        neighbourhoods_df: pd.DataFrame | None = None,
        return_interval: bool = True,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
            id, predicted_price, lower_bound, upper_bound (optional)
        """
        if reviews_df is None:
            reviews_df = pd.DataFrame(columns=["listing_id", "date"])
        if neighbourhoods_df is None:
            neighbourhoods_df = pd.DataFrame(columns=["neighbourhood_group", "neighbourhood"])

        features_df = self.engineer.transform(listings_df, reviews_df, neighbourhoods_df)
        X = self.preprocessor.transform_new(features_df)
        y_log = self.predictor.predict(X)
        y_price = self.preprocessor.inverse_transform_target(y_log)

        result = pd.DataFrame({
            "id": listings_df["id"].values,
            "predicted_price": np.round(y_price, 2),
        })

        if return_interval:
            # Simple uncertainty estimate: ±1 std of residuals (calibrate from eval)
            residual_std_path = self.output_dir / "residual_std.json"
            if residual_std_path.exists():
                with open(residual_std_path) as f:
                    std = json.load(f).get("residual_std_usd", 30.0)
            else:
                std = 30.0  # fallback
            result["lower_bound"] = np.round(np.clip(y_price - 1.96 * std, 0, None), 2)
            result["upper_bound"] = np.round(y_price + 1.96 * std, 2)

        return result


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Run inference on new listings")
    parser.add_argument("--input",      required=True,  help="Path to new listings CSV")
    parser.add_argument("--output",     default="predictions.csv")
    parser.add_argument("--output_dir", default="outputs/")
    parser.add_argument("--data_dir",   default="data/")
    args = parser.parse_args()

    engine = PriceInferenceEngine(args.output_dir, args.data_dir)
    new_listings = pd.read_csv(args.input)
    preds = engine.predict(new_listings)
    preds.to_csv(args.output, index=False)
    logger.info(f"Predictions saved to {args.output}")
    print(preds.describe())