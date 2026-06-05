"""
LA Airbnb Price Prediction Pipeline
====================================
Production-level ML pipeline with:
- Data loading & validation
- Feature engineering (geo, temporal, text, host, neighbourhood)
- Model training with hyperparameter tuning
- Evaluation & explainability
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.preprocessor import Preprocessor
from src.model import PricePredictor
from src.evaluator import Evaluator
from src.utils import setup_logging, save_artifact

def run_pipeline(
    data_dir: str = "data/",
    output_dir: str = "outputs/",
    model_type: str = "xgboost",        # xgboost | lightgbm | ensemble
    tune_hyperparams: bool = False,
    log_transform_target: bool = True,
):
    logger = setup_logging()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    logger.info("Loading data …")
    loader = DataLoader(data_dir)
    listings_df, reviews_df, neighbourhoods_df, geo_df = loader.load_all()
    logger.info(f"  listings: {listings_df.shape}  reviews: {reviews_df.shape}")

    # ── 2. Feature Engineering ───────────────────────────────────────────────
    logger.info("Engineering features …")
    engineer = FeatureEngineer(geo_df)
    features_df = engineer.transform(listings_df, reviews_df, neighbourhoods_df)
    logger.info(f"  feature matrix: {features_df.shape}")

    # ── 3. Preprocessing ─────────────────────────────────────────────────────
    logger.info("Preprocessing …")
    prep = Preprocessor(log_transform_target=log_transform_target)
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prep.fit_transform(features_df)
    logger.info(f"  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")
    save_artifact(prep, output_dir, "preprocessor.pkl")

    # ── 4. Train ─────────────────────────────────────────────────────────────
    logger.info(f"Training [{model_type}] …")
    predictor = PricePredictor(
        model_type=model_type,
        feature_names=feature_names,
        tune_hyperparams=tune_hyperparams,
    )
    predictor.fit(X_train, y_train, X_val, y_val)
    save_artifact(predictor, output_dir, "model.pkl")

    # ── 5. Evaluate ──────────────────────────────────────────────────────────
    logger.info("Evaluating …")
    evaluator = Evaluator(log_transform_target=log_transform_target)
    metrics = evaluator.evaluate(predictor, X_test, y_test, feature_names, output_dir)

    logger.info("=== TEST METRICS ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    logger.info(f"Pipeline complete. Artifacts saved to '{output_dir}'")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/")
    parser.add_argument("--output_dir", default="outputs/")
    parser.add_argument("--model",      default="ensemble", choices=["xgboost","lightgbm","ensemble"])
    parser.add_argument("--tune",       action="store_true")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_type=args.model,
        tune_hyperparams=args.tune,
    )