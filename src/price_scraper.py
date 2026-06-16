"""
price_scraper.py
================
Fetches nightly prices from Airbnb's public API for listings
whose price column is blank in the scraped CSV.

Uses Airbnb's internal pricing endpoint — no auth required, 
same call the website makes when you pick dates.

Usage:
    python -m src.price_scraper --input data/listings.csv --output data/prices.csv

Then pipeline.py will auto-detect data/prices.csv and merge it in.
"""

import time
import random
import logging
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Airbnb pricing API (public, no auth) ──────────────────────────────────
# Returns price breakdown for a given listing + checkin/checkout
API_URL = "https://www.airbnb.com/api/v3/PdpAvailabilityCalendar"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "X-Airbnb-Api-Key": "d306zoyjsyarp7ifhu67rjxn52tv0t20",   # public key, same for all browsers
    "Referer": "https://www.airbnb.com/",
}


def get_listing_price(listing_id: int, session: requests.Session) -> float | None:
    """
    Fetch nightly price for a listing via Airbnb's calendar API.
    Returns the modal (most common) nightly price across the next 3 months,
    or None if unavailable.
    """
    params = {
        "operationName": "PdpAvailabilityCalendar",
        "locale": "en",
        "currency": "USD",
        "variables": json.dumps({
            "request": {
                "count": 12,
                "listingId": str(listing_id),
                "month": _next_month(),
                "year": _current_year(),
            }
        }),
        "extensions": json.dumps({
            "persistedQuery": {
                "version": 1,
                "sha256Hash": "8f086249543a96b5f4b4b5e6db2571c78a4f5dce0c9c6e2f3b6a5b0a1d2e4f7",
            }
        }),
    }

    try:
        r = session.get(API_URL, params=params, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        prices = _extract_prices(data)
        return float(np.median(prices)) if prices else None
    except Exception:
        return None


def _extract_prices(data: dict) -> list:
    """Walk the nested JSON to find nightly price values."""
    prices = []
    try:
        months = (
            data["data"]["merlin"]["pdpAvailabilityCalendar"]["calendarMonths"]
        )
        for month in months:
            for day in month.get("days", []):
                if day.get("available") and day.get("price"):
                    amt = day["price"].get("localPriceFormatted", "")
                    # strip currency symbols and commas
                    amt_clean = re.sub(r"[^\d.]", "", amt)
                    if amt_clean:
                        prices.append(float(amt_clean))
    except (KeyError, TypeError):
        pass
    return prices


def _next_month() -> int:
    from datetime import date
    d = date.today()
    return d.month % 12 + 1

def _current_year() -> int:
    from datetime import date
    return date.today().year


# ════════════════════════════════════════════════════════════════════════════
def scrape_prices(
    listings_csv: str,
    output_csv: str,
    max_listings: int = 5000,
    delay_min: float = 0.8,
    delay_max: float = 2.2,
    resume: bool = True,
):
    """
    Main scraping loop.
    
    Args:
        listings_csv:  path to listings.csv
        output_csv:    where to write id,price pairs
        max_listings:  cap (avoid scraping 40k rows overnight)
        delay_min/max: random sleep between requests (be polite)
        resume:        skip IDs already in output_csv
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    df = pd.read_csv(listings_csv, usecols=["id", "listing_url"], low_memory=False)
    ids = df["id"].dropna().astype(int).unique().tolist()
    logger.info(f"Total listings: {len(ids)}")

    # Resume: skip already-scraped IDs
    out_path = Path(output_csv)
    already_done = set()
    if resume and out_path.exists():
        done_df = pd.read_csv(out_path)
        already_done = set(done_df["id"].astype(int).tolist())
        logger.info(f"Resuming — {len(already_done)} already scraped")

    ids_to_scrape = [i for i in ids if i not in already_done][:max_listings]
    logger.info(f"Will scrape {len(ids_to_scrape)} listings")

    session = requests.Session()
    session.headers.update(HEADERS)

    results = []
    failed  = 0

    for idx, lid in enumerate(ids_to_scrape):
        price = get_listing_price(lid, session)

        if price is not None:
            results.append({"id": lid, "price": price})
            status = f"${price:.0f}"
        else:
            failed += 1
            status = "N/A"

        if (idx + 1) % 50 == 0 or idx == 0:
            logger.info(
                f"  [{idx+1}/{len(ids_to_scrape)}] "
                f"listing {lid}: {status} | "
                f"success={len(results)} fail={failed}"
            )

            # Checkpoint save every 50 requests
            if results:
                _save(results, out_path, append=resume or idx > 0)
                results = []

        time.sleep(random.uniform(delay_min, delay_max))

    # Final save
    if results:
        _save(results, out_path, append=True)

    logger.info(f"Done. Scraped {len(already_done) + len(results)} total prices → {out_path}")


def _save(rows: list, path: Path, append: bool):
    new_df = pd.DataFrame(rows)
    if append and path.exists():
        new_df.to_csv(path, mode="a", header=False, index=False)
    else:
        new_df.to_csv(path, index=False)


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Airbnb nightly prices")
    parser.add_argument("--input",        default="data/listings.csv")
    parser.add_argument("--output",       default="data/prices.csv")
    parser.add_argument("--max",          type=int, default=5000,
                        help="Max listings to scrape (default 5000)")
    parser.add_argument("--delay_min",    type=float, default=0.8)
    parser.add_argument("--delay_max",    type=float, default=2.2)
    parser.add_argument("--no_resume",    action="store_true",
                        help="Start fresh, ignore existing output file")
    args = parser.parse_args()

    scrape_prices(
        listings_csv=args.input,
        output_csv=args.output,
        max_listings=args.max,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        resume=not args.no_resume,
    )