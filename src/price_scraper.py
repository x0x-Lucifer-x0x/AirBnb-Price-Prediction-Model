# price_scraper.py

import argparse
import csv
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
import undetected_chromedriver as uc

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = PROJECT_ROOT / "data" / "listings.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "listing_price.csv"

PRICE_REGEX = re.compile(r"[₹$€£]\s?[\d,]+")


STATUS_SUCCESS = "SUCCESS"
STATUS_PRICE_NOT_FOUND = "PRICE_NOT_FOUND"
STATUS_LOGIN_REQUIRED = "LOGIN_REQUIRED"
STATUS_NO_AVAILABLE_DATES = "NO_AVAILABLE_DATES"
STATUS_CALENDAR_NOT_FOUND = "CALENDAR_NOT_FOUND"
STATUS_CHECK_AVAILABILITY_NOT_FOUND = "CHECK_AVAILABILITY_NOT_FOUND"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_ACCESS_DENIED = "ACCESS_DENIED"
STATUS_CAPTCHA = "CAPTCHA"
STATUS_LISTING_REMOVED = "LISTING_REMOVED"
STATUS_UNKNOWN_ERROR = "UNKNOWN_ERROR"


def random_sleep(a=2.5, b=5.5):
    time.sleep(random.uniform(a, b))


def create_driver():
    options = uc.ChromeOptions()

    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-notifications")
    options.add_argument("--no-default-browser-check")
    options.add_argument("--disable-popup-blocking")

    driver = uc.Chrome(
        options=options,
        use_subprocess=True
    )

    return driver


def human_scroll(driver):
    try:
        driver.execute_script(
            "window.scrollTo(0, arguments[0]);",
            random.randint(500, 1500)
        )
        random_sleep(1, 2)
    except:
        pass


def detect_block_page(driver):
    source = driver.page_source.lower()

    if "captcha" in source:
        return STATUS_CAPTCHA

    if "access denied" in source:
        return STATUS_ACCESS_DENIED

    if "log in" in source and "continue with email" in source:
        return STATUS_LOGIN_REQUIRED

    if "this listing is no longer available" in source:
        return STATUS_LISTING_REMOVED

    return None


def click_check_availability(driver):
    possible_xpaths = [
        "//button[contains(., 'Check availability')]",
        "//span[contains(., 'Check availability')]",
        "//*[@aria-label='Check availability']"
    ]

    for xpath in possible_xpaths:
        try:
            btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, xpath))
            )

            driver.execute_script(
                "arguments[0].scrollIntoView({block:'center'});",
                btn
            )

            random_sleep(1, 2)

            try:
                btn.click()
            except:
                driver.execute_script("arguments[0].click();", btn)

            random_sleep(2, 3)
            return True

        except:
            continue

    return False


def find_available_dates(driver):
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (
                    By.CSS_SELECTOR,
                    "[data-testid*='calendar-day']"
                )
            )
        )

        available = driver.find_elements(
            By.CSS_SELECTOR,
            "[data-testid*='calendar-day'][data-is-day-blocked='false']"
        )

        if len(available) < 2:
            return False

        driver.execute_script(
            "arguments[0].click();",
            available[0]
        )

        random_sleep(1, 2)

        checkout_index = min(3, len(available) - 1)

        driver.execute_script(
            "arguments[0].click();",
            available[checkout_index]
        )

        random_sleep(4, 6)

        return True

    except:
        return False


def extract_price(driver):
    selectors = [
        "span.u1opajno"
    ]

    for selector in selectors:
        try:
            elements = driver.find_elements(
                By.CSS_SELECTOR,
                selector
            )

            for element in elements:
                text = element.text.strip()

                if PRICE_REGEX.search(text):
                    return text

        except:
            pass

    try:
        body_text = driver.find_element(
            By.TAG_NAME,
            "body"
        ).text

        match = PRICE_REGEX.search(body_text)

        if match:
            return match.group()

    except:
        pass

    try:
        source = driver.page_source

        match = PRICE_REGEX.search(source)

        if match:
            return match.group()

    except:
        pass

    return None


def scrape_listing(driver, listing_id, url):
    try:
        driver.get(url)

        random_sleep(4, 7)

        human_scroll(driver)

        block_status = detect_block_page(driver)

        if block_status:
            return {
                "id": listing_id,
                "listing_url": url,
                "price": "",
                "status": "FAILED",
                "remark": block_status
            }

        if not click_check_availability(driver):
            return {
                "id": listing_id,
                "listing_url": url,
                "price": "",
                "status": "FAILED",
                "remark": STATUS_CHECK_AVAILABILITY_NOT_FOUND
            }

        if not find_available_dates(driver):
            return {
                "id": listing_id,
                "listing_url": url,
                "price": "",
                "status": "FAILED",
                "remark": STATUS_NO_AVAILABLE_DATES
            }

        price = extract_price(driver)

        if not price:
            return {
                "id": listing_id,
                "listing_url": url,
                "price": "",
                "status": "FAILED",
                "remark": STATUS_PRICE_NOT_FOUND
            }

        numeric_price = re.sub(r"[^\d]", "", price)

        return {
            "id": listing_id,
            "listing_url": url,
            "price": numeric_price,
            "status": STATUS_SUCCESS,
            "remark": ""
        }

    except TimeoutException:
        return {
            "id": listing_id,
            "listing_url": url,
            "price": "",
            "status": "FAILED",
            "remark": STATUS_TIMEOUT
        }

    except Exception as e:
        return {
            "id": listing_id,
            "listing_url": url,
            "price": "",
            "status": "FAILED",
            "remark": f"{STATUS_UNKNOWN_ERROR}: {str(e)[:150]}"
        }


def write_header():
    if OUTPUT_CSV.exists():
        return

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "listing_url",
                "price",
                "status",
                "remark"
            ]
        )

        writer.writeheader()


def append_result(result):
    with open(
        OUTPUT_CSV,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "listing_url",
                "price",
                "status",
                "remark"
            ]
        )

        writer.writerow(result)


def run_test(url):
    driver = create_driver()

    try:
        listing_id = url.rstrip("/").split("/")[-1]

        result = scrape_listing(
            driver,
            listing_id,
            url
        )

        print("\nRESULT")
        print("-" * 50)

        for k, v in result.items():
            print(f"{k}: {v}")

    finally:
        input("\nPress ENTER to close browser...")
        driver.quit()


def run_all():
    if not INPUT_CSV.exists():
        print(f"\nMissing file:\n{INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)

    if "id" not in df.columns:
        print("Missing 'id' column")
        sys.exit(1)

    if "listing_url" not in df.columns:
        print("Missing 'listing_url' column")
        sys.exit(1)

    write_header()

    driver = create_driver()

    success = 0
    failed = 0

    total = len(df)

    try:
        for idx, row in enumerate(df.itertuples(index=False), start=1):

            listing_id = row.id
            listing_url = row.listing_url

            result = scrape_listing(
                driver,
                listing_id,
                listing_url
            )

            append_result(result)

            if result["status"] == STATUS_SUCCESS:
                success += 1

                print(
                    f"[{idx}/{total}] "
                    f"ID={listing_id} "
                    f"SUCCESS "
                    f"₹{result['price']}"
                )

            else:
                failed += 1

                print(
                    f"[{idx}/{total}] "
                    f"ID={listing_id} "
                    f"FAILED "
                    f"{result['remark']}"
                )

            print(
                f"Success={success} "
                f"Failed={failed}"
            )

            random_sleep(3, 8)

    finally:
        driver.quit()

    print("\nDONE")
    print("-" * 50)
    print(f"Total   : {total}")
    print(f"Success : {success}")
    print(f"Failed  : {failed}")
    print(f"Output  : {OUTPUT_CSV}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        help="Test a single Airbnb listing URL"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all listings from CSV"
    )

    args = parser.parse_args()

    if args.test:
        run_test(args.test)

    elif args.all:
        run_all()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
