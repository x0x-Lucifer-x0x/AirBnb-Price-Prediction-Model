"""
Data Loader
===========
Price recovery priority chain:
  1. price column in listings.csv          (populated in some scrapes)
  2. data/prices.csv                       (output of price_scraper.py)
  3. calendar.csv / calendar.csv.gz        (if price column is populated there)
  4. neighbourhood + room_type median      (partial imputation from above sources)
  5. RuntimeError with clear fix steps
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

    def load_all(self):
        listings       = self._load_listings()
        reviews        = self._load_reviews()
        neighbourhoods = self._load_neighbourhoods()
        geo            = self._load_geojson()
        return listings, reviews, neighbourhoods, geo

    # ================================================================== #
    def _load_listings(self) -> pd.DataFrame:
        path = self._find("listings.csv")
        df   = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()
        before = len(df)

        # ── normalise column names ─────────────────────────────────────
        _rename = {
            "neighbourhood_cleansed":       "neighbourhood",
            "neighbourhood_group_cleansed": "neighbourhood_group",
        }
        for old, new in _rename.items():
            if old in df.columns and new not in df.columns:
                df.rename(columns={old: new}, inplace=True)

        # ── Step 1: price column in listings.csv ──────────────────────
        if "price" not in df.columns:
            df["price"] = np.nan
        else:
            df["price"] = self._parse_money(df["price"])
        n_missing = int(df["price"].isna().sum())
        logger.info(f"  listings.csv: {before} rows, {n_missing} missing price after step 1")

        # ── Step 2: prices.csv from scraper ───────────────────────────
        if n_missing > 0:
            scraped = self._load_scraped_prices()
            if scraped is not None:
                df["price"] = df["price"].fillna(df["id"].map(scraped))
                n_filled = n_missing - int(df["price"].isna().sum())
                logger.info(f"  Step 2 (prices.csv): filled {n_filled} prices")
                n_missing = int(df["price"].isna().sum())

        # ── Step 3: calendar.csv ──────────────────────────────────────
        if n_missing > 0:
            cal_prices = self._load_calendar_prices()
            if cal_prices is not None:
                df["price"] = df["price"].fillna(df["id"].map(cal_prices))
                n_filled = n_missing - int(df["price"].isna().sum())
                logger.info(f"  Step 3 (calendar): filled {n_filled} prices")
                n_missing = int(df["price"].isna().sum())

        # ── Step 4: median imputation within neighbourhood+room_type ──
        if 0 < n_missing < before:
            df["price"] = self._group_median_fill(df, ["neighbourhood", "room_type"])
            df["price"] = self._group_median_fill(df, ["neighbourhood_group", "room_type"])
            df["price"] = self._group_median_fill(df, ["room_type"])
            n_filled = n_missing - int(df["price"].isna().sum())
            logger.info(f"  Step 4 (group median): filled {n_filled} prices")

        # ── Step 5: fail with instructions ───────────────────────────
        if df["price"].isna().all():
            raise RuntimeError(self._no_price_message())

        # ── Filter invalid prices ─────────────────────────────────────
        valid = df["price"].notna() & (df["price"] > 0) & (df["price"] < 25_000)
        df = df[valid].reset_index(drop=True)
        logger.info(f"  Final: {len(df)}/{before} rows with valid price")

        self._validate_columns(df, self.REQUIRED_LISTING_COLS, "listings")
        return df

    # ================================================================== #
    def _load_scraped_prices(self) -> "pd.Series | None":
        """Load prices.csv produced by price_scraper.py → listing_id: price."""
        p = self.data_dir / "prices.csv"
        if not p.exists():
            return None
        try:
            df = pd.read_csv(p)
            df.columns = df.columns.str.strip().str.lower()
            if "id" not in df.columns or "price" not in df.columns:
                logger.warning("  prices.csv missing 'id' or 'price' column")
                return None
            df["price"] = self._parse_money(df["price"])
            series = df.dropna(subset=["price"]).set_index("id")["price"]
            logger.info(f"  prices.csv: {len(series)} scraped prices loaded")
            return series
        except Exception as e:
            logger.warning(f"  Could not load prices.csv: {e}")
            return None

    def _load_calendar_prices(self) -> "pd.Series | None":
        """Load calendar.csv[.gz] → listing_id: median price."""
        for fname in ("calendar.csv", "calendar.csv.gz"):
            p = self.data_dir / fname
            if not p.exists():
                continue
            logger.info(f"  Loading {fname}…")
            try:
                cal = pd.read_csv(
                    p,
                    usecols=lambda c: c.strip().lower() in (
                        "listing_id", "price", "adjusted_price", "available"
                    ),
                    low_memory=False,
                )
                cal.columns = cal.columns.str.strip().str.lower()

                # Try adjusted_price first, fall back to price
                price_col = None
                for col in ("adjusted_price", "price"):
                    if col in cal.columns:
                        s = self._parse_money(cal[col])
                        if s.notna().sum() > 0:
                            price_col = col
                            cal["_price"] = s
                            break

                if price_col is None:
                    logger.warning(f"  {fname}: both price columns are blank")
                    continue

                if "available" in cal.columns:
                    cal = cal[cal["available"].astype(str).str.lower() == "t"]

                cal = cal[cal["_price"].notna() & (cal["_price"] > 0)]
                if cal.empty:
                    logger.warning(f"  {fname}: no valid prices after filtering")
                    continue

                result = cal.groupby("listing_id")["_price"].median()
                logger.info(f"  {fname}: {len(result)} listing prices recovered")
                return result

            except Exception as e:
                logger.warning(f"  Could not parse {fname}: {e}")
        return None

    def _load_reviews(self) -> pd.DataFrame:
        for fname in ("reviews.csv", "reviews.csv.gz"):
            p = self.data_dir / fname
            if p.exists():
                df = pd.read_csv(p, low_memory=False, parse_dates=["date"])
                df.columns = df.columns.str.strip().str.lower()
                logger.info(f"  reviews: {len(df)} rows from {fname}")
                return df
        logger.warning("  No reviews file — review features will be skipped")
        return pd.DataFrame(columns=["listing_id", "date"])

    def _load_neighbourhoods(self) -> pd.DataFrame:
        df = pd.read_csv(self._find("neighbourhoods.csv"))
        df.columns = df.columns.str.strip().str.lower()
        return df

    def _load_geojson(self) -> dict:
        with open(self._find("neighbourhoods.geojson")) as f:
            return json.load(f)

    # ================================================================== #
    #  Helpers
    # ================================================================== #
    @staticmethod
    def _parse_money(s: pd.Series) -> pd.Series:
        return (
            s.astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .str.strip()
            .replace({"nan": np.nan, "none": np.nan, "n/a": np.nan, "": np.nan})
            .astype(float)
        )

    @staticmethod
    def _group_median_fill(df: pd.DataFrame, cols: list) -> pd.Series:
        present = [c for c in cols if c in df.columns]
        if not present:
            return df["price"]
        return df["price"].fillna(df.groupby(present)["price"].transform("median"))

    def _find(self, filename: str) -> Path:
        p = self.data_dir / filename
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")
        return p

    @staticmethod
    def _validate_columns(df: pd.DataFrame, required: list, name: str):
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"  [{name}] missing columns: {missing}")

    @staticmethod
    def _no_price_message() -> str:
        return """
╔══════════════════════════════════════════════════════════════╗
║  PRICE DATA MISSING — model cannot be trained                ║
╠══════════════════════════════════════════════════════════════╣
║  All price columns in listings.csv and calendar.csv are      ║
║  blank. This is a known issue with some Inside Airbnb        ║
║  scrapes where prices are stripped before publishing.        ║
║                                                              ║
║  SOLUTION — run the price scraper first:                     ║
║                                                              ║
║    python -m src.price_scraper \\                            ║
║        --input  data/listings.csv \\                         ║
║        --output data/prices.csv \\                           ║
║        --max    5000                                         ║
║                                                              ║
║  This fetches live nightly prices from Airbnb's API          ║
║  for up to 5000 listings (~2-3 hours with polite delays).    ║
║  Then re-run pipeline.py — prices.csv is picked up auto.     ║
╚══════════════════════════════════════════════════════════════╝
"""