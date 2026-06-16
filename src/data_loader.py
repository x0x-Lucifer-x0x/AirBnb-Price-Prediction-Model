"""
Data Loader
===========
Handles reading and basic validation of all raw source files.

Price recovery strategy (in order):
  1. price column in listings.csv (if populated)
  2. calendar.csv / calendar.csv.gz  ← median nightly price per listing
  3. RuntimeError with clear instructions
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

    # ================================================================== #
    def _load_listings(self) -> pd.DataFrame:
        path = self._find("listings.csv")
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()
        before = len(df)

        # ── normalise column names (detailed schema vs summary) ────────
        if "neighbourhood_cleansed" in df.columns and "neighbourhood" not in df.columns:
            df.rename(columns={"neighbourhood_cleansed": "neighbourhood"}, inplace=True)
        if "neighbourhood_group_cleansed" in df.columns and "neighbourhood_group" not in df.columns:
            df.rename(columns={"neighbourhood_group_cleansed": "neighbourhood_group"}, inplace=True)

        # ── Step 1: clean price column ─────────────────────────────────
        if "price" not in df.columns:
            df["price"] = np.nan
        else:
            df["price"] = self._parse_price_series(df["price"])

        n_missing = int(df["price"].isna().sum())
        logger.info(f"  listings raw: {before} rows, {n_missing} missing price")

        # ── Step 2: fill from calendar if needed ──────────────────────
        if n_missing > 0:
            cal_prices = self._load_calendar_prices()
            if cal_prices is not None:
                df["price"] = df["price"].fillna(df["id"].map(cal_prices))
                filled = n_missing - int(df["price"].isna().sum())
                logger.info(f"  Filled {filled} prices from calendar.csv")

        # ── Step 3: fill remaining via neighbourhood+room_type median ──
        still_missing = int(df["price"].isna().sum())
        if still_missing > 0 and still_missing < before:
            # Only impute if we have SOME real prices to compute medians from
            df["price"] = self._impute_by_group(df, ["neighbourhood", "room_type"], "price")
            df["price"] = self._impute_by_group(df, ["neighbourhood_group", "room_type"], "price")
            df["price"] = self._impute_by_group(df, ["room_type"], "price")
            after_impute = int(df["price"].isna().sum())
            logger.info(f"  Imputed {still_missing - after_impute} prices via neighbourhood median")

        # ── Step 4: hard fail if EVERYTHING is blank ──────────────────
        if df["price"].isna().all():
            raise RuntimeError(
                "\n"
                "═══════════════════════════════════════════════════════\n"
                "  PRICE DATA MISSING — cannot train model\n"
                "═══════════════════════════════════════════════════════\n"
                "  This scrape does not include nightly prices.\n"
                "  Fix: add calendar.csv.gz to your data/ folder.\n"
                "\n"
                "  Download it from Inside Airbnb:\n"
                "  http://insideairbnb.com/get-the-data/\n"
                "  → Los Angeles → 'calendar.csv.gz'\n"
                "  → gunzip it → rename to calendar.csv\n"
                "  → place in your data/ folder\n"
                "═══════════════════════════════════════════════════════\n"
            )

        # ── Step 5: filter out invalid prices ─────────────────────────
        valid = df["price"].notna() & (df["price"] > 0) & (df["price"] < 25_000)
        df = df[valid].reset_index(drop=True)
        logger.info(f"  listings final: {len(df)}/{before} rows with valid price")

        self._validate_columns(df, self.REQUIRED_LISTING_COLS, "listings")
        return df

    # ================================================================== #
    def _load_calendar_prices(self) -> "pd.Series | None":
        """
        Load calendar.csv[.gz] and return Series:  listing_id → median price.
        Calendar has one row per listing per day with a 'price' column.
        """
        for fname in ("calendar.csv", "calendar.csv.gz"):
            p = self.data_dir / fname
            if not p.exists():
                continue

            logger.info(f"  Loading calendar from {fname} (may take a moment)…")
            try:
                # Only read what we need — calendar files are large (~1 GB)
                cal = pd.read_csv(
                    p,
                    usecols=lambda c: c in ("listing_id", "price", "available"),
                    low_memory=False,
                )
                cal.columns = cal.columns.str.strip().str.lower()

                if "price" not in cal.columns or "listing_id" not in cal.columns:
                    logger.warning(f"  {fname}: missing expected columns, skipping")
                    continue

                # Only use days where listing is available (price is real offer price)
                if "available" in cal.columns:
                    cal = cal[cal["available"].astype(str).str.lower() == "t"]

                cal["price"] = self._parse_price_series(cal["price"])
                cal = cal[cal["price"].notna() & (cal["price"] > 0) & (cal["price"] < 25_000)]

                median_prices = cal.groupby("listing_id")["price"].median()
                logger.info(f"  Calendar: got prices for {len(median_prices)} listings")
                return median_prices

            except Exception as e:
                logger.warning(f"  Could not parse {fname}: {e}")
                continue

        logger.info("  No calendar file found in data/ — skipping calendar price fill")
        return None

    # ================================================================== #
    def _load_reviews(self) -> pd.DataFrame:
        for fname in ("reviews.csv", "reviews.csv.gz"):
            p = self.data_dir / fname
            if p.exists():
                df = pd.read_csv(p, low_memory=False, parse_dates=["date"])
                df.columns = df.columns.str.strip().str.lower()
                logger.info(f"  reviews loaded from {fname}: {len(df)} rows")
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

    # ================================================================== #
    #  Helpers
    # ================================================================== #
    @staticmethod
    def _parse_price_series(s: pd.Series) -> pd.Series:
        return (
            s.astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .str.strip()
            .replace({"nan": np.nan, "none": np.nan, "": np.nan})
            .astype(float)
        )

    @staticmethod
    def _impute_by_group(df: pd.DataFrame, group_cols: list, target: str) -> pd.Series:
        """Fill NaN in target with group median. Only groups that exist in df."""
        cols_present = [c for c in group_cols if c in df.columns]
        if not cols_present:
            return df[target]
        medians = df.groupby(cols_present)[target].transform("median")
        return df[target].fillna(medians)

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