"""
Preprocessor
============
Handles train/val/test splitting, missing value imputation,
scaling, and target transformation.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, List

logger = logging.getLogger(__name__)

# Columns to EXCLUDE from model features
EXCLUDE_COLS = [
    "id", "name", "host_id", "host_name", "host_profile_id",
    "neighbourhood", "neighbourhood_group",  # encoded versions kept
    "latitude", "longitude",                 # geo_cluster / dist features kept
    "room_type",                             # ordinal + dummies kept
    "price",                                 # target
    "last_review", "last_review_dt",
    "listing_id",                            # from reviews merge
    "last_review_date", "first_review_date",
    "license",
    "price_pct_in_neigh",                    # leaks target rank → remove for strict eval
]


class Preprocessor:
    def __init__(
        self,
        val_size: float = 0.10,
        test_size: float = 0.10,
        log_transform_target: bool = True,
        random_state: int = 42,
    ):
        self.val_size = val_size
        self.test_size = test_size
        self.log_transform_target = log_transform_target
        self.random_state = random_state

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = RobustScaler()        # robust to price outliers
        self.feature_names: List[str] = []

    # ------------------------------------------------------------------ #
    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray, List[str]]:

        y = df["price"].values.astype(float)
        if self.log_transform_target:
            y = np.log1p(y)

        # ── select feature columns ────────────────────────────────────
        drop = [c for c in EXCLUDE_COLS if c in df.columns]
        X_df = df.drop(columns=drop)

        # keep only numeric columns
        X_df = X_df.select_dtypes(include=[np.number])
        self.feature_names = X_df.columns.tolist()
        X = X_df.values.astype(float)

        logger.info(f"  {len(self.feature_names)} numeric features selected")

        # ── stratified split by log-price quintile ────────────────────
        quintile = pd.qcut(y, q=5, labels=False, duplicates="drop")
        X_tmp, X_test, y_tmp, y_test, q_tmp, _ = train_test_split(
            X, y, quintile,
            test_size=self.test_size,
            stratify=quintile,
            random_state=self.random_state,
        )
        val_frac = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp,
            test_size=val_frac,
            stratify=q_tmp,
            random_state=self.random_state,
        )

        # ── impute then scale (fit on train only) ─────────────────────
        X_train = self.imputer.fit_transform(X_train)
        X_val   = self.imputer.transform(X_val)
        X_test  = self.imputer.transform(X_test)

        X_train = self.scaler.fit_transform(X_train)
        X_val   = self.scaler.transform(X_val)
        X_test  = self.scaler.transform(X_test)

        logger.info(
            f"  Split → train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test, self.feature_names

    # ------------------------------------------------------------------ #
    def transform_new(self, df: pd.DataFrame) -> np.ndarray:
        """Transform unseen data at inference time."""
        drop = [c for c in EXCLUDE_COLS if c in df.columns]
        X_df = df.drop(columns=drop, errors="ignore")
        X_df = X_df.select_dtypes(include=[np.number])

        # align columns
        for col in self.feature_names:
            if col not in X_df.columns:
                X_df[col] = 0.0
        X_df = X_df[self.feature_names]

        X = self.imputer.transform(X_df.values.astype(float))
        return self.scaler.transform(X)

    # ------------------------------------------------------------------ #
    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        if self.log_transform_target:
            return np.expm1(y)
        return y