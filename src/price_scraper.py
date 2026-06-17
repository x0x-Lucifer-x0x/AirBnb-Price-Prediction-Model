"""
price_scraper.py  —  Selenium Airbnb nightly price scraper
===========================================================
Verified against actual DOM (June 2026):

  Day cells     : <td role="button" aria-disabled="..." aria-label="...">
                    <div data-testid="calendar-day-DD/MM/YYYY"
                         data-is-day-blocked="true|false">
  Available     : aria-disabled="false" AND label contains "Select as check-in date"
  Blocked       : aria-disabled="true"  OR  data-is-day-blocked="true"
  Skip (no co)  : label contains "has no eligible checkout date"
  Next month    : button[aria-label="Move forward to change to the next month."]
  Price span    : span.u1opajno  (e.g. ₹1,95,198)
  Close button  : button[data-testid="availability-calendar-save"]

Usage:
    # Test a single listing (browser window visible):
    python -m src.price_scraper --test --url https://www.airbnb.com/rooms/2708

    # Scrape all listings (headless):
    python -m src.price_scraper --input data/listings.csv --output data/prices.csv --max 3000
"""

import re
import time
import random
import logging
import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Currency → USD conversion ─────────────────────────────────────────────────
CURRENCY_RATES = {
    "₹": 1 / 84.0,
    "€": 1.08,
    "£": 1.27,
    "$": 1.0,
    "A$": 0.65,
    "C$": 0.73,
}


def _to_usd(amount: float, symbol: str) -> float:
    return round(amount * CURRENCY_RATES.get(symbol, 1.0), 2)


def _detect_currency(text: str) -> str:
    for sym in ("₹", "€", "£", "A$", "C$", "$"):
        if sym in text:
            return sym
    return "$"


# ══════════════════════════════════════════════════════════════════════════════
def _build_driver(headless: bool = True):
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager

    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--window-size=1400,900")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=opts,
    )
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


# ══════════════════════════════════════════════════════════════════════════════
class AirbnbPriceScraper:
    """
    Exact selectors verified from live DOM inspection.
    """

    # ── Verified selectors ────────────────────────────────────────────────────
    # "Add dates for prices" text on page (opens calendar)
    XPATH_OPEN_CAL = '//*[contains(text(),"Add dates for prices")]'

    # Calendar day cells: <td role="button" ...>
    # Available for check-in = aria-disabled false + label has "Select as check-in date"
    CSS_ALL_DAYS   = 'td[role="button"]'

    # Next month arrow — EXACT aria-label from DOM
    CSS_NEXT_MONTH = 'button[aria-label="Move forward to change to the next month."]'

    # Price span — short stable class confirmed from DOM
    CSS_PRICE      = "span.u1opajno"

    # Close calendar button
    CSS_CLOSE_CAL  = 'button[data-testid="availability-calendar-save"]'

    # ── Input fields to type dates directly (fallback) ────────────────────────
    ID_CHECKIN  = "checkIn-book_it"
    ID_CHECKOUT = "checkOut-book_it"

    def __init__(self, driver):
        self.driver = driver

    # ------------------------------------------------------------------ #
    def get_price(self, listing_url: str, min_nights: int = 1) -> float | None:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        try:
            self.driver.get(listing_url)
            time.sleep(random.uniform(3.0, 5.0))   # full JS render
            self._dismiss_popups()

            # ── Open calendar ──────────────────────────────────────────
            if not self._open_calendar():
                logger.debug("  Could not open calendar")
                return None
            time.sleep(1.5)

            # ── Find first valid check-in date ─────────────────────────
            checkin_td, checkin_date = self._find_first_checkin()
            if checkin_td is None:
                logger.debug("  No valid check-in date found in visible months")
                return None

            self._click_day(checkin_td)
            time.sleep(1.2)

            # ── Find checkout date = checkin + min_nights ──────────────
            checkout_date = checkin_date + timedelta(days=max(min_nights, 1))
            checkout_td   = self._find_day_td(checkout_date)

            # If checkout date not found or blocked, scroll forward and try +1 day
            for extra in range(15):
                candidate = checkout_date + timedelta(days=extra)
                td = self._find_day_td(candidate)
                if td is not None and self._td_is_available(td):
                    checkout_td   = td
                    checkout_date = candidate
                    break
            else:
                checkout_td = None

            if checkout_td is None:
                logger.debug("  No valid checkout date found")
                return None

            self._click_day(checkout_td)
            time.sleep(2.0)   # wait for price to update

            # ── Read price ─────────────────────────────────────────────
            nights = (checkout_date - checkin_date).days
            if nights == 0:
                return None

            raw_price, symbol = self._read_price()
            if raw_price is None:
                logger.debug("  Could not read price from span.u1opajno")
                return None

            nightly_local = raw_price / nights
            nightly_usd   = _to_usd(nightly_local, symbol)
            logger.debug(
                f"  {symbol}{raw_price:.0f} / {nights} nights "
                f"= {symbol}{nightly_local:.0f}/night "
                f"→ ${nightly_usd:.0f} USD"
            )

            # Close calendar cleanly before next listing
            self._close_calendar()
            return nightly_usd

        except Exception as e:
            logger.debug(f"  Exception: {e}")
            return None

    # ================================================================== #
    #  Calendar navigation helpers
    # ================================================================== #
    def _open_calendar(self) -> bool:
        from selenium.webdriver.common.by import By

        # Try the "Add dates for prices" text
        try:
            el = self.driver.find_element(By.XPATH, self.XPATH_OPEN_CAL)
            el.click()
            return True
        except Exception:
            pass

        # Try check-in input field
        try:
            el = self.driver.find_element(By.ID, self.ID_CHECKIN)
            el.click()
            return True
        except Exception:
            pass

        return False

    def _close_calendar(self):
        from selenium.webdriver.common.by import By
        try:
            btn = self.driver.find_element(By.CSS_SELECTOR, self.CSS_CLOSE_CAL)
            btn.click()
            time.sleep(0.5)
        except Exception:
            pass

    def _next_month(self) -> bool:
        """Click the next-month arrow. Returns True if clicked."""
        from selenium.webdriver.common.by import By
        try:
            btn = self.driver.find_element(By.CSS_SELECTOR, self.CSS_NEXT_MONTH)
            btn.click()
            time.sleep(0.8)
            return True
        except Exception:
            return False

    # ================================================================== #
    #  Day cell helpers  (all operate on <td role="button"> elements)
    # ================================================================== #
    def _all_day_tds(self):
        from selenium.webdriver.common.by import By
        return self.driver.find_elements(By.CSS_SELECTOR, self.CSS_ALL_DAYS)

    def _td_is_checkin_available(self, td) -> bool:
        """
        A valid check-in cell:
          aria-disabled="false"
          aria-label contains "Select as check-in date"
          (excludes "has no eligible checkout date" cells)
        """
        label = td.get_attribute("aria-label") or ""
        disabled = td.get_attribute("aria-disabled")
        return (
            disabled == "false"
            and "Select as check-in date" in label
        )

    def _td_is_available(self, td) -> bool:
        """Any non-blocked cell (including checkout-only cells)."""
        disabled = td.get_attribute("aria-disabled")
        label    = td.get_attribute("aria-label") or ""
        return disabled == "false" and "Unavailable" not in label

    def _td_to_date(self, td) -> date | None:
        """
        Parse date from inner div's data-testid="calendar-day-DD/MM/YYYY".
        """
        from selenium.webdriver.common.by import By
        try:
            div = td.find_element(By.CSS_SELECTOR, "div[data-testid^='calendar-day-']")
            testid = div.get_attribute("data-testid")  # e.g. "calendar-day-11/08/2026"
            date_str = testid.replace("calendar-day-", "")  # "11/08/2026"
            day, month, year = date_str.split("/")
            return date(int(year), int(month), int(day))
        except Exception:
            return None

    def _find_first_checkin(self) -> "tuple[object|None, date|None]":
        """
        Scan visible calendar (up to 4 months forward) for the first
        valid check-in cell on or after today.
        """
        today = date.today()
        for _ in range(4):          # try up to 4 months
            for td in self._all_day_tds():
                if not self._td_is_checkin_available(td):
                    continue
                d = self._td_to_date(td)
                if d is not None and d >= today:
                    return td, d
            # No valid date in current view → go to next month
            if not self._next_month():
                break
        return None, None

    def _find_day_td(self, target: date):
        """
        Find the <td> for a specific date in the currently rendered calendar.
        Uses data-testid="calendar-day-DD/MM/YYYY" on the inner div.
        Scrolls forward up to 3 months if not found.
        """
        from selenium.webdriver.common.by import By

        testid = f"calendar-day-{target.day:02d}/{target.month:02d}/{target.year}"
        for _ in range(3):
            try:
                div = self.driver.find_element(
                    By.CSS_SELECTOR, f'div[data-testid="{testid}"]'
                )
                # The parent <td> is the clickable element
                return div.find_element(By.XPATH, "..")
            except Exception:
                pass
            if not self._next_month():
                break
        return None

    def _click_day(self, td):
        """Click a day cell, scrolling into view first."""
        self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", td)
        time.sleep(0.3)
        try:
            td.click()
        except Exception:
            # Fallback: JS click
            self.driver.execute_script("arguments[0].click();", td)

    # ================================================================== #
    #  Price reading
    # ================================================================== #
    def _read_price(self) -> "tuple[float|None, str]":
        """
        Read total price from span.u1opajno  (e.g. "₹1,95,198").
        Returns (numeric_amount, currency_symbol).
        """
        from selenium.webdriver.common.by import By

        time.sleep(1.0)   # let price update after checkout click

        try:
            spans = self.driver.find_elements(By.CSS_SELECTOR, self.CSS_PRICE)
            for span in spans:
                text = span.text.strip()
                if not text:
                    continue
                symbol = _detect_currency(text)
                # Strip everything except digits and dots
                numeric_str = re.sub(r"[^\d.]", "", text)
                if numeric_str:
                    return float(numeric_str), symbol
        except Exception as e:
            logger.debug(f"  _read_price error: {e}")

        return None, "$"

    # ================================================================== #
    def _dismiss_popups(self):
        from selenium.webdriver.common.by import By
        for xpath in [
            '//button[@aria-label="Close"]',
            '//button[contains(text(),"Decline")]',
            '//button[contains(text(),"Not now")]',
        ]:
            try:
                for btn in self.driver.find_elements(By.XPATH, xpath)[:1]:
                    btn.click()
                    time.sleep(0.3)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
#  Main scraping loop
# ══════════════════════════════════════════════════════════════════════════════
def scrape_prices(
    listings_csv: str,
    output_csv: str,
    max_listings: int = 3000,
    delay_min: float = 2.5,
    delay_max: float = 5.0,
    headless: bool = True,
    resume: bool = True,
    batch_size: int = 25,
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    df = pd.read_csv(listings_csv, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    for col in ("listing_url", "id", "minimum_nights"):
        if col not in df.columns:
            raise ValueError(f"listings.csv missing column: {col}")

    df["minimum_nights"] = (
        pd.to_numeric(df["minimum_nights"], errors="coerce").fillna(1).astype(int)
    )
    rows = df[["id", "listing_url", "minimum_nights"]].dropna(subset=["listing_url"])

    # ── Resume ────────────────────────────────────────────────────────
    out_path = Path(output_csv)
    done_ids: set = set()
    if resume and out_path.exists():
        done_df = pd.read_csv(out_path)
        done_ids = set(done_df["id"].astype(int).tolist())
        logger.info(f"Resuming — {len(done_ids)} already scraped")

    todo = rows[~rows["id"].astype(int).isin(done_ids)].head(max_listings)
    logger.info(f"Listings to scrape: {len(todo)}")
    if todo.empty:
        logger.info("Nothing left to scrape.")
        return

    driver  = _build_driver(headless=headless)
    scraper = AirbnbPriceScraper(driver)
    results = []
    success = fail = 0

    try:
        for i, (_, row) in enumerate(todo.iterrows()):
            lid     = int(row["id"])
            url     = str(row["listing_url"])
            min_nts = int(row["minimum_nights"])

            price = scraper.get_price(url, min_nights=min_nts)

            if price and 1 < price < 25_000:
                results.append({"id": lid, "price": price})
                success += 1
                status = f"${price:.0f}/night"
            else:
                fail += 1
                status = "FAILED"

            logger.info(
                f"[{i+1:>4}/{len(todo)}] id={lid:<8} {status:<18} "
                f"min_nights={min_nts:<4} ok={success} fail={fail}"
            )

            if results and (i + 1) % batch_size == 0:
                _save(results, out_path)
                results = []

            time.sleep(random.uniform(delay_min, delay_max))

            # Restart browser every 150 listings
            if (i + 1) % 150 == 0:
                logger.info("Restarting browser…")
                driver.quit()
                time.sleep(3)
                driver  = _build_driver(headless=headless)
                scraper = AirbnbPriceScraper(driver)

    except KeyboardInterrupt:
        logger.info("Interrupted — saving progress…")
    finally:
        if results:
            _save(results, out_path)
        try:
            driver.quit()
        except Exception:
            pass

    logger.info(
        f"\n{'='*50}\n"
        f"  Done.  Success={success}  Fail={fail}\n"
        f"  Total saved: {len(done_ids) + success}\n"
        f"  Output: {out_path}\n"
        f"{'='*50}"
    )


def _save(rows: list, path: Path):
    new_df = pd.DataFrame(rows)
    if path.exists():
        new_df.to_csv(path, mode="a", header=False, index=False)
    else:
        new_df.to_csv(path, index=False)
    logger.info(f"  ✓ Saved {len(rows)} rows → {path}")


# ══════════════════════════════════════════════════════════════════════════════
def test_single(url: str, min_nights: int = 1, headless: bool = False):
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    driver  = _build_driver(headless=headless)
    scraper = AirbnbPriceScraper(driver)
    try:
        logger.info(f"Testing: {url}")
        price = scraper.get_price(url, min_nights=min_nights)
        if price:
            logger.info(f"✓ SUCCESS → ${price:.2f} USD/night")
        else:
            logger.warning("✗ FAILED — could not extract price")
    finally:
        if not headless:
            input("Press Enter to close browser…")
        driver.quit()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       default="data/listings.csv")
    parser.add_argument("--output",      default="data/prices.csv")
    parser.add_argument("--max",         type=int,   default=3000)
    parser.add_argument("--delay_min",   type=float, default=2.5)
    parser.add_argument("--delay_max",   type=float, default=5.0)
    parser.add_argument("--no_headless", action="store_true")
    parser.add_argument("--no_resume",   action="store_true")
    parser.add_argument("--batch_size",  type=int,   default=25)
    parser.add_argument("--test",        action="store_true")
    parser.add_argument("--url",         default=None)
    parser.add_argument("--min_nights",  type=int,   default=1)
    args = parser.parse_args()

    if args.test:
        if not args.url:
            parser.error("--test requires --url")
        test_single(args.url, min_nights=args.min_nights, headless=not args.no_headless)
    else:
        scrape_prices(
            listings_csv=args.input,
            output_csv=args.output,
            max_listings=args.max,
            delay_min=args.delay_min,
            delay_max=args.delay_max,
            headless=not args.no_headless,
            resume=not args.no_resume,
            batch_size=args.batch_size,
        )