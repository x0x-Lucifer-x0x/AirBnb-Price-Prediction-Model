"""
Feature Engineering
===================
Derives every signal we can extract from listings, reviews, neighbourhoods,
and the GeoJSON geometry.

Feature groups
--------------
1.  Core listing attributes  (room type, min nights, availability …)
2.  Host features            (host listing count, multi-listing flag …)
3.  Review / sentiment       (count, recency, velocity, avg-score proxies)
4.  Geographic               (neighbourhood group dummies, lat/lon clusters,
                              distance to key LA landmarks)
5.  Neighbourhood statistics (median price per neighbourhood, density …)
6.  Temporal                 (last-review recency, seasonal proxies)
7.  Interaction terms        (room_type × availability, min_nights × room_type …)
"""

import logging
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ── Key LA landmarks for distance features ──────────────────────────────────
LANDMARKS = {
    "downtown_la":    (34.0522, -118.2437),
    "lax_airport":    (33.9425, -118.4081),
    "santa_monica_pier": (34.0100, -118.4965),
    "hollywood_sign": (34.1341, -118.3215),
    "venice_beach":   (33.9850, -118.4695),
    "beverly_hills":  (34.0736, -118.4004),
    "universal_studios": (34.1381, -118.3534),
    "dodger_stadium": (34.0739, -118.2400),
}


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in kilometres."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


class FeatureEngineer:
    def __init__(self, geo_data: dict, n_geo_clusters: int = 20):
        self.geo_data = geo_data
        self.n_geo_clusters = n_geo_clusters
        self._neigh_stats: dict = {}          # filled during fit phase
        self._geo_kmeans: KMeans | None = None

    # ================================================================== #
    def transform(
        self,
        listings: pd.DataFrame,
        reviews: pd.DataFrame,
        neighbourhoods: pd.DataFrame,
    ) -> pd.DataFrame:

        df = listings.copy()
        df = enrich_with_detailed_columns(df)

        df = self._clean_core(df)
        df = self._host_features(df)
        df = self._review_features(df, reviews)
        df = self._geo_features(df)
        df = self._neighbourhood_stats(df)
        df = self._temporal_features(df)
        df = self._interaction_features(df)

        logger.info(f"  FeatureEngineer: {df.shape[1]} columns after engineering")
        return df

    # ================================================================== #
    #  1. Core listing cleaning
    # ================================================================== #
    def _clean_core(self, df: pd.DataFrame) -> pd.DataFrame:
        # Room type → ordinal  (private < shared < hotel < entire)
        room_order = {
            "Shared room": 0,
            "Private room": 1,
            "Hotel room": 2,
            "Entire home/apt": 3,
        }
        df["room_type_ord"] = df["room_type"].map(room_order).fillna(1)

        # One-hot room type
        rt_dummies = pd.get_dummies(df["room_type"], prefix="rt", drop_first=False)
        df = pd.concat([df, rt_dummies], axis=1)

        # Availability ratios
        df["availability_rate"] = df["availability_365"] / 365.0
        df["is_always_available"] = (df["availability_365"] == 365).astype(int)

        # Min nights capped + log
        df["minimum_nights_capped"] = df["minimum_nights"].clip(upper=365)
        df["log_min_nights"] = np.log1p(df["minimum_nights_capped"])

        # Long-term vs short-term rental flag
        df["is_long_term"] = (df["minimum_nights"] >= 28).astype(int)
        df["is_medium_term"] = ((df["minimum_nights"] >= 7) & (df["minimum_nights"] < 28)).astype(int)

        # Has license
        df["has_license"] = df["license"].notna().astype(int) if "license" in df.columns else 0

        return df

    # ================================================================== #
    #  2. Host features
    # ================================================================== #
    def _host_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["log_host_listings"] = np.log1p(df["calculated_host_listings_count"].fillna(1))
        df["is_multi_host"] = (df["calculated_host_listings_count"] > 1).astype(int)
        df["is_super_host"] = (df["calculated_host_listings_count"] >= 5).astype(int)

        # Host market share per neighbourhood
        host_neigh = (
            df.groupby(["neighbourhood", "host_id"])["id"]
            .count()
            .reset_index()
            .rename(columns={"id": "host_listings_in_neigh"})
        )
        neigh_total = (
            df.groupby("neighbourhood")["id"]
            .count()
            .reset_index()
            .rename(columns={"id": "total_listings_in_neigh"})
        )
        host_neigh = host_neigh.merge(neigh_total, on="neighbourhood")
        host_neigh["host_market_share"] = (
            host_neigh["host_listings_in_neigh"] / host_neigh["total_listings_in_neigh"]
        )
        df = df.merge(
            host_neigh[["neighbourhood", "host_id", "host_market_share"]],
            on=["neighbourhood", "host_id"],
            how="left",
        )
        df["host_market_share"] = df["host_market_share"].fillna(0)
        return df

    # ================================================================== #
    #  3. Review features
    # ================================================================== #
    def _review_features(self, df: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
        if reviews.empty or "listing_id" not in reviews.columns:
            df["review_count_actual"] = df["number_of_reviews"].fillna(0)
            df["has_reviews"] = (df["number_of_reviews"] > 0).astype(int)
            df["log_reviews"] = np.log1p(df["number_of_reviews"].fillna(0))
            df["reviews_per_month_filled"] = df["reviews_per_month"].fillna(0)
            df["days_since_last_review"] = 730  # unknown → 2 years
            df["review_recency_score"] = 0.0
            df["ltm_review_ratio"] = 0.0
            return df

        today = reviews["date"].max()

        agg = reviews.groupby("listing_id").agg(
            review_count_actual=("date", "count"),
            last_review_date=("date", "max"),
            first_review_date=("date", "min"),
        ).reset_index()

        agg["days_since_last_review"] = (today - agg["last_review_date"]).dt.days
        agg["listing_age_days"] = (today - agg["first_review_date"]).dt.days.clip(lower=1)
        agg["review_velocity"] = agg["review_count_actual"] / agg["listing_age_days"] * 30  # per month

        df = df.merge(agg, left_on="id", right_on="listing_id", how="left")

        df["review_count_actual"] = df["review_count_actual"].fillna(0)
        df["days_since_last_review"] = df["days_since_last_review"].fillna(730)
        df["review_velocity"] = df["review_velocity"].fillna(0)
        df["has_reviews"] = (df["review_count_actual"] > 0).astype(int)
        df["log_reviews"] = np.log1p(df["review_count_actual"])
        df["reviews_per_month_filled"] = df["reviews_per_month"].fillna(0)

        # Recency score: 1 = reviewed today, 0 = never/very old
        max_days = 730
        df["review_recency_score"] = (
            1 - (df["days_since_last_review"].clip(upper=max_days) / max_days)
        )

        # LTM ratio (how active in last 12 months relative to all time)
        df["ltm_review_ratio"] = np.where(
            df["review_count_actual"] > 0,
            df["number_of_reviews_ltm"].fillna(0) / df["review_count_actual"].clip(lower=1),
            0,
        )
        return df

    # ================================================================== #
    #  4. Geographic features
    # ================================================================== #
    def _geo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        lat = df["latitude"].values
        lon = df["longitude"].values

        # Distance to each landmark
        for name, (lm_lat, lm_lon) in LANDMARKS.items():
            df[f"dist_{name}_km"] = [
                haversine_km(la, lo, lm_lat, lm_lon)
                for la, lo in zip(lat, lon)
            ]

        # Minimum distance to any beach / tourist hub
        beach_cols = ["dist_santa_monica_pier_km", "dist_venice_beach_km"]
        tourist_cols = ["dist_hollywood_sign_km", "dist_universal_studios_km", "dist_beverly_hills_km"]
        df["min_beach_dist_km"]   = df[beach_cols].min(axis=1)
        df["min_tourist_dist_km"] = df[tourist_cols].min(axis=1)
        df["log_dist_downtown"]   = np.log1p(df["dist_downtown_la_km"])

        # Geographic clusters via K-Means (spatial market segmentation)
        coords = df[["latitude", "longitude"]].fillna(df[["latitude", "longitude"]].mean())

        if len(coords) == 0:
            raise ValueError(
                "DataFrame is empty before KMeans geo-clustering. "
                "This usually means all rows were dropped during price filtering. "
                "Check that your listings.csv contains a 'price' column with valid values. "
                "The detailed listings.csv.gz (not the summary listings.csv) is required."
            )

        n_clusters = min(self.n_geo_clusters, len(coords))
        if self._geo_kmeans is None:
            self._geo_kmeans = KMeans(
                n_clusters=n_clusters, random_state=42, n_init=10
            )
            self._geo_kmeans.fit(coords)

        df["geo_cluster"] = self._geo_kmeans.predict(coords)
        geo_dummies = pd.get_dummies(df["geo_cluster"], prefix="geo_cluster")
        df = pd.concat([df, geo_dummies], axis=1)

        # Neighbourhood group one-hot
        ng_dummies = pd.get_dummies(df["neighbourhood_group"], prefix="ng")
        df = pd.concat([df, ng_dummies], axis=1)

        return df

    # ================================================================== #
    #  5. Neighbourhood statistics
    # ================================================================== #
    def _neighbourhood_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Target-encode neighbourhood with smoothing to avoid leakage.
        Uses the entire dataset statistics (not train-only); a proper
        cross-validated version is handled in Preprocessor.
        """
        global_median = df["price"].median()
        k = 10  # smoothing strength

        neigh_stats = (
            df.groupby("neighbourhood")["price"]
            .agg(["median", "mean", "std", "count"])
            .reset_index()
        )
        neigh_stats.columns = [
            "neighbourhood", "neigh_price_median",
            "neigh_price_mean", "neigh_price_std", "neigh_listing_count",
        ]
        # Smoothed mean encoding
        neigh_stats["neigh_price_smoothed"] = (
            (neigh_stats["neigh_price_mean"] * neigh_stats["neigh_listing_count"]
             + global_median * k)
            / (neigh_stats["neigh_listing_count"] + k)
        )
        neigh_stats["neigh_price_std"] = neigh_stats["neigh_price_std"].fillna(0)
        neigh_stats["neigh_log_count"] = np.log1p(neigh_stats["neigh_listing_count"])

        df = df.merge(neigh_stats, on="neighbourhood", how="left")

        # Percentile rank of each listing within its neighbourhood
        df["price_pct_in_neigh"] = df.groupby("neighbourhood")["price"].rank(pct=True)

        return df

    # ================================================================== #
    #  6. Temporal features
    # ================================================================== #
    def _temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "last_review" in df.columns:
            df["last_review_dt"] = pd.to_datetime(df["last_review"], errors="coerce")
            ref = pd.Timestamp.now()
            df["days_since_last_review_listing"] = (
                (ref - df["last_review_dt"]).dt.days.fillna(1460)  # 4 years default
            )
            df["last_review_month"] = df["last_review_dt"].dt.month.fillna(0)
            df["last_review_year"]  = df["last_review_dt"].dt.year.fillna(0)
            # Proxy for high-season (summer / winter holidays)
            df["last_review_peak_season"] = df["last_review_month"].isin([6,7,8,12]).astype(int)
        return df

    # ================================================================== #
    #  7. Interaction features
    # ================================================================== #
    def _interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["room_x_avail"] = df["room_type_ord"] * df["availability_rate"]
        df["room_x_min_nights"] = df["room_type_ord"] * df["log_min_nights"]
        df["review_x_recency"] = df["log_reviews"] * df["review_recency_score"]
        df["beach_x_room"] = df["min_beach_dist_km"] * df["room_type_ord"]
        df["host_listings_x_avail"] = df["log_host_listings"] * df["availability_rate"]
        df["downtown_x_room"] = df["log_dist_downtown"] * df["room_type_ord"]
        return df


# ══════════════════════════════════════════════════════════════════════════════
#  Extra features unlocked by the detailed listings file
# ══════════════════════════════════════════════════════════════════════════════
def enrich_with_detailed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Called inside FeatureEngineer.transform() when detailed columns exist.
    Safe to call on summary-only data — all ops are guarded by column checks.
    """

    # ── Capacity & physical attributes ───────────────────────────────
    if "accommodates" in df.columns:
        df["accommodates"] = pd.to_numeric(df["accommodates"], errors="coerce").fillna(1)
        df["log_accommodates"] = np.log1p(df["accommodates"])

    if "bedrooms" in df.columns:
        df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce").fillna(1)

    if "beds" in df.columns:
        df["beds"] = pd.to_numeric(df["beds"], errors="coerce").fillna(1)
        df["beds_per_person"] = df["beds"] / df.get("accommodates", pd.Series(1, index=df.index)).clip(lower=1)

    if "bathrooms" in df.columns:
        df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce").fillna(1)
    elif "bathrooms_text" in df.columns:
        # e.g. "1 shared bath", "1.5 baths" → extract numeric
        df["bathrooms"] = (
            df["bathrooms_text"]
            .astype(str)
            .str.extract(r"(\d+\.?\d*)")[0]
            .astype(float)
            .fillna(1)
        )
        df["is_shared_bath"] = df["bathrooms_text"].str.lower().str.contains("shared").astype(int)

    if "property_type" in df.columns:
        # broad buckets: entire_home, private_room, hotel, shared
        df["is_entire_home_prop"] = df["property_type"].str.lower().str.contains("entire").astype(int)
        df["is_hotel_prop"] = df["property_type"].str.lower().str.contains("hotel|boutique").astype(int)
        df["is_guest_suite"] = df["property_type"].str.lower().str.contains("guest suite|guesthouse").astype(int)

    # ── Host quality signals ──────────────────────────────────────────
    if "host_is_superhost" in df.columns:
        df["host_is_superhost_flag"] = (df["host_is_superhost"].astype(str).str.lower() == "t").astype(int)

    if "host_identity_verified" in df.columns:
        df["host_verified"] = (df["host_identity_verified"].astype(str).str.lower() == "t").astype(int)

    if "host_response_rate" in df.columns:
        df["host_response_rate_num"] = (
            df["host_response_rate"]
            .astype(str).str.replace("%", "").str.strip()
            .replace({"N/A": np.nan, "nan": np.nan})
            .astype(float) / 100
        )

    if "host_acceptance_rate" in df.columns:
        df["host_acceptance_rate_num"] = (
            df["host_acceptance_rate"]
            .astype(str).str.replace("%", "").str.strip()
            .replace({"N/A": np.nan, "nan": np.nan})
            .astype(float) / 100
        )

    if "hosts_time_as_host_years" in df.columns:
        df["host_experience_years"] = pd.to_numeric(df["hosts_time_as_host_years"], errors="coerce").fillna(0)

    # ── Review scores (6 dimensions) ─────────────────────────────────
    score_cols = [
        "review_scores_rating", "review_scores_accuracy",
        "review_scores_cleanliness", "review_scores_checkin",
        "review_scores_communication", "review_scores_location",
        "review_scores_value",
    ]
    present_scores = [c for c in score_cols if c in df.columns]
    for col in present_scores:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if present_scores:
        df["avg_review_score"] = df[present_scores].mean(axis=1)
        df["min_review_score"] = df[present_scores].min(axis=1)
        df["review_score_spread"] = df[present_scores].max(axis=1) - df[present_scores].min(axis=1)
        # location score relative to overall (premium location signal)
        if "review_scores_location" in df.columns and "review_scores_rating" in df.columns:
            df["location_vs_overall"] = df["review_scores_location"] - df["review_scores_rating"]

    # ── Booking / calendar signals ────────────────────────────────────
    if "instant_bookable" in df.columns:
        df["instant_bookable_flag"] = (df["instant_bookable"].astype(str).str.lower() == "t").astype(int)

    if "has_availability" in df.columns:
        df["has_availability_flag"] = (df["has_availability"].astype(str).str.lower() == "t").astype(int)

    if "availability_30" in df.columns:
        df["avail_30_rate"] = pd.to_numeric(df["availability_30"], errors="coerce") / 30
    if "availability_60" in df.columns:
        df["avail_60_rate"] = pd.to_numeric(df["availability_60"], errors="coerce") / 60
    if "availability_90" in df.columns:
        df["avail_90_rate"] = pd.to_numeric(df["availability_90"], errors="coerce") / 90

    # ── Amenity features ─────────────────────────────────────────────
    if "amenities" in df.columns:
        amen = df["amenities"].astype(str).str.lower()
        df["has_pool"]        = amen.str.contains("pool").astype(int)
        df["has_hot_tub"]     = amen.str.contains("hot tub|jacuzzi").astype(int)
        df["has_gym"]         = amen.str.contains("gym|exercise equipment").astype(int)
        df["has_ev_charger"]  = amen.str.contains("ev charger").astype(int)
        df["has_beach_access"]= amen.str.contains("beach access|beachfront").astype(int)
        df["has_fireplace"]   = amen.str.contains("fireplace").astype(int)
        df["has_workspace"]   = amen.str.contains("dedicated workspace").astype(int)
        df["has_ac"]          = amen.str.contains("air conditioning").astype(int)
        df["has_washer"]      = amen.str.contains(r"\bwasher\b").astype(int)
        df["has_parking"]     = amen.str.contains("parking").astype(int)
        df["has_pets_allowed"]= amen.str.contains("pets allowed").astype(int)
        df["has_breakfast"]   = amen.str.contains("breakfast").astype(int)
        df["has_self_checkin"]= amen.str.contains("self check-in|lockbox|keypad").astype(int)
        # amenity count as a richness proxy
        df["amenity_count"] = (
            df["amenities"].astype(str)
            .str.count('","')
            .add(1)
            .clip(upper=100)
        )

    # ── Estimated revenue (additional target signal, useful as feature) ──
    if "estimated_revenue_l365d" in df.columns:
        df["log_estimated_revenue"] = np.log1p(
            pd.to_numeric(df["estimated_revenue_l365d"], errors="coerce").fillna(0)
        )

    if "estimated_occupancy_l365d" in df.columns:
        df["occupancy_rate"] = (
            pd.to_numeric(df["estimated_occupancy_l365d"], errors="coerce").fillna(0) / 365
        ).clip(0, 1)

    return df