"""
Data Loader
===========
Handles reading and basic validation of all raw source files.
Supports both the summary listings.csv and the detailed listings.csv.gz
from Inside Airbnb. When the price column is blank (common in detailed files),
price is reconstructed from estimated_revenue_l365d + estimated_occupancy_l365d.
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
        listings       = self._load_listings()
        reviews        = self._load_reviews()
        neighbourhoods = self._load_neighbourhoods()
        geo            = self._load_geojson()
        return listings, reviews, neighbourhoods, geo

    # ------------------------------------------------------------------ #
    def _load_listings(self) -> pd.DataFrame:
        path = self._find("listings.csv")
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()

        # ── normalise column names (detailed schema differs from summary) ──
        # detailed: neighbourhood_cleansed / neighbourhood_group_cleansed
        if "neighbourhood_cleansed" in df.columns and "neighbourhood" not in df.columns:
            df.rename(columns={"neighbourhood_cleansed": "neighbourhood"}, inplace=True)
        if "neighbourhood_group_cleansed" in df.columns and "neighbourhood_group" not in df.columns:
            df.rename(columns={"neighbourhood_group_cleansed": "neighbourhood_group"}, inplace=True)

        before = len(df)

        # ── clean price column if present ─────────────────────────────
        if "price" in df.columns:
            df["price"] = (
                df["price"]
                .astype(str)
                .str.replace(r"[\$,]", "", regex=True)
                .str.strip()
                .replace({"nan": np.nan, "": np.nan})
                .astype(float)
            )
        else:
            df["price"] = np.nan

        n_missing = int(df["price"].isna().sum())

        # ── reconstruct price from revenue/occupancy when blank ───────
        if n_missing > 0:
            reconstructed = self._reconstruct_price(df)
            df["price"] = df["price"].fillna(reconstructed)
            n_filled = n_missing - int(df["price"].isna().sum())
            if n_filled > 0:
                logger.info(
                    f"  Reconstructed price for {n_filled}/{n_missing} rows "
                    "from estimated_revenue_l365d / estimated_occupancy_l365d"
                )

        # ── final validation ──────────────────────────────────────────
        still_missing = int(df["price"].isna().sum())
        if still_missing == before:
            raise RuntimeError(
                "Could not determine price for ANY listing.\n"
                "The 'price' column is blank AND estimated_revenue_l365d / "
                "estimated_occupancy_l365d are also missing or zero.\n"
                "Ensure you have the correct detailed listings file from Inside Airbnb."
            )

        valid_mask = df["price"].notna() & (df["price"] > 0) & (df["price"] < 25_000)
        df = df[valid_mask]
        logger.info(
            f"  listings: kept {len(df)}/{before} rows "
            f"(dropped {before - len(df)} with missing/invalid price)"
        )

        self._validate_columns(df, self.REQUIRED_LISTING_COLS, "listings")
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _reconstruct_price(df: pd.DataFrame) -> pd.Series:
        """
        Derive nightly price from Inside Airbnb estimated metrics.

        Inside Airbnb provides:
          estimated_revenue_l365d    – estimated annual revenue  ($)
          estimated_occupancy_l365d  – estimated occupied nights in last 365 days

        price ≈ revenue / occupied_nights

        Fallback chain:
          1. revenue / occupancy          (best)
          2. revenue / availability_365   (when occupancy is 0 / missing)
          3. revenue / 90                 (assume ~25% occupancy as last resort)
        """
        result = pd.Series(np.nan, index=df.index)

        def to_num(col):
            if col not in df.columns:
                return pd.Series(np.nan, index=df.index)
            return pd.to_numeric(
                df[col].astype(str).str.replace(r"[\$,]", "", regex=True),
                errors="coerce",
            )

        revenue   = to_num("estimated_revenue_l365d")
        occupancy = to_num("estimated_occupancy_l365d")
        avail     = to_num("availability_365")

        # Method 1: revenue / occupancy nights
        m1 = revenue.notna() & (revenue > 0) & occupancy.notna() & (occupancy > 0)
        result[m1] = (revenue[m1] / occupancy[m1]).round(2)

        # Method 2: revenue / availability days
        m2 = result.isna() & revenue.notna() & (revenue > 0) & avail.notna() & (avail > 0)
        result[m2] = (revenue[m2] / avail[m2]).round(2)

        # Method 3: revenue / 90  (rough fallback)
        m3 = result.isna() & revenue.notna() & (revenue > 0)
        result[m3] = (revenue[m3] / 90).round(2)

        return result

    # ------------------------------------------------------------------ #
    def _load_reviews(self) -> pd.DataFrame:
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