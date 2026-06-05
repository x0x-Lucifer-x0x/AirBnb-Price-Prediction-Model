"""
Data Loader
===========
Handles reading and basic validation of all raw source files.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoader:
    REQUIRED_LISTING_COLS = [
        "id", "neighbourhood_group", "neighbourhood",
        "latitude", "longitude", "room_type",
        "price", "minimum_nights", "number_of_reviews",
        "reviews_per_month", "calculated_host_listings_count",
        "availability_365", "number_of_reviews_ltm",
    ]

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------ #
    def load_all(self):
        listings      = self._load_listings()
        reviews       = self._load_reviews()
        neighbourhoods = self._load_neighbourhoods()
        geo           = self._load_geojson()
        return listings, reviews, neighbourhoods, geo

    # ------------------------------------------------------------------ #
    def _load_listings(self) -> pd.DataFrame:
        path = self._find("listings.csv")
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()

        # ── price cleaning ────────────────────────────────────────────
        if "price" in df.columns:
            df["price"] = (
                df["price"]
                .astype(str)
                .str.replace(r"[\$,]", "", regex=True)
                .str.strip()
                .replace({"nan": np.nan, "": np.nan})
                .astype(float)
            )
        # drop rows with no price (target unknown)
        before = len(df)
        df = df[df["price"].notna() & (df["price"] > 0) & (df["price"] < 25_000)]
        logger.info(f"  listings: dropped {before - len(df)} rows (missing/invalid price)")

        self._validate_columns(df, self.REQUIRED_LISTING_COLS, "listings")
        return df.reset_index(drop=True)

    def _load_reviews(self) -> pd.DataFrame:
        # Support both summary reviews.csv and detailed reviews.csv.gz
        for fname in ("reviews.csv", "reviews.csv.gz"):
            p = self.data_dir / fname
            if p.exists():
                df = pd.read_csv(p, low_memory=False, parse_dates=["date"])
                df.columns = df.columns.str.strip().str.lower()
                logger.info(f"  reviews loaded from {fname}")
                return df
        logger.warning("  No reviews file found – review features will be skipped")
        return pd.DataFrame(columns=["listing_id", "date"])

    def _load_neighbourhoods(self) -> pd.DataFrame:
        path = self._find("neighbourhoods.csv")
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        return df

    def _load_geojson(self) -> dict:
        path = self._find("neighbourhoods.geojson")
        with open(path) as f:
            geo = json.load(f)
        return geo

    # ------------------------------------------------------------------ #
    def _find(self, filename: str) -> Path:
        p = self.data_dir / filename
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")
        return p

    @staticmethod
    def _validate_columns(df: pd.DataFrame, required: list, name: str):
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"  [{name}] missing expected columns: {missing}")