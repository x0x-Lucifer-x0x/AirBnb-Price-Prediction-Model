"""
run_scraper.py
==============
Convenience wrapper — edit the config below and run:

    python run_scraper.py

This will populate data/prices.csv, then you can run:

    python pipeline.py
"""

from src.price_scraper import scrape_prices

# ── CONFIGURE HERE ────────────────────────────────────────────────────────────
INPUT_CSV   = "data/listings.csv"
OUTPUT_CSV  = "data/prices.csv"

MAX_LISTINGS = 3000       # How many listings to scrape total
                           # 3000 listings ≈ 3-5 hours at polite speed
                           # Safe to Ctrl+C and restart — it resumes automatically

HEADLESS     = True        # False = shows Chrome window (useful for debugging)
DELAY_MIN    = 2.5         # Seconds between listings (min) — don't go below 1.5
DELAY_MAX    = 5.0         # Seconds between listings (max)
BATCH_SIZE   = 25          # Save to disk every N listings
RESUME       = True        # True = skip already-scraped IDs
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════════════════╗
║  Airbnb Price Scraper                                ║
╠══════════════════════════════════════════════════════╣
║  Input  : {INPUT_CSV:<42}                            ║
║  Output : {OUTPUT_CSV:<42}                           ║
║  Max    : {str(MAX_LISTINGS):<42}                    ║
║  Headless: {str(HEADLESS):<41}                       ║
╠══════════════════════════════════════════════════════╣
║  Safe to Ctrl+C — progress is saved every            ║
║  {BATCH_SIZE} listings to {OUTPUT_CSV:<35}           ║
║  Re-run to resume from where you stopped.            ║
╚══════════════════════════════════════════════════════╝
    """)

    scrape_prices(
        listings_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
        max_listings=MAX_LISTINGS,
        headless=HEADLESS,
        delay_min=DELAY_MIN,
        delay_max=DELAY_MAX,
        batch_size=BATCH_SIZE,
        resume=RESUME,
    )

    print(f"\n✓ Done! Now run:  python pipeline.py")