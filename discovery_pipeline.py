import argparse
import logging
import os
import json
import time
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv

from src.core.config import AppConfig
from src.core.logging_config import setup_logging
from src.scraper.scraper_utils import find_links, normalize_url
from src.utils.helpers import sanitize_filename_component
import asyncio
import csv
import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from urllib.parse import urljoin

# Setup logger
logger = logging.getLogger(__name__)

def load_discovery_profile(profile_name: str) -> Optional[Dict[str, Any]]:
    """Loads the specified discovery profile from the discovery_profiles directory."""
    logger.info(f"Attempting to load discovery profile: {profile_name}")
    # Construct path relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    profile_path = os.path.join(script_dir, 'discovery_profiles', f"{profile_name}.json")

    if not os.path.exists(profile_path):
        logger.error(f"Discovery profile file not found at: {profile_path}")
        return None

    try:
        with open(profile_path, 'r', encoding='utf-8') as f:
            profile_data = json.load(f)
            logger.info(f"Successfully loaded discovery profile '{profile_name}'.")
            return profile_data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from profile file {profile_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading profile {profile_path}: {e}", exc_info=True)
        return None

def run_discovery_pipeline(profile: Dict[str, Any], app_config: AppConfig, output_dir: str, limit: int):
    """
    Main logic for the discovery pipeline.
    - Iterates through listing pages.
    - Navigates to company detail pages.
    - Extracts structured data based on profile selectors.
    - Saves the structured data to a CSV file.
    """
    logger.info(f"Running discovery pipeline for source: {profile.get('source_name')}")
    
    start_urls = profile.get("start_urls", [])
    listing_selectors = profile.get("listing_page_selectors", {})
    detail_extractors = profile.get("detail_page_extractors", {})
    max_pages = profile.get("max_pages_to_crawl_per_start_url", 1)

    if not all([start_urls, listing_selectors, detail_extractors]):
        logger.error("Profile is missing one or more required keys: 'start_urls', 'listing_page_selectors', 'detail_page_extractors'.")
        return

    all_discovered_companies_data = []
    processed_detail_pages = set()

    async def discovery_main():
        # Create a subdirectory for saving HTML content for debugging
        html_debug_dir = os.path.join(output_dir, "html_debug")
        os.makedirs(html_debug_dir, exist_ok=True)
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            for start_url in start_urls:
                if limit > 0 and len(all_discovered_companies_data) >= limit: break
                
                logger.info(f"Processing start URL: {start_url}")
                current_url: Optional[str] = start_url
                pages_crawled = 0

                while current_url and pages_crawled < max_pages:
                    if limit > 0 and len(all_discovered_companies_data) >= limit: break
                    
                    try:
                        await page.goto(current_url, timeout=60000, wait_until='networkidle')
                        logger.info(f"Successfully navigated to listing page: {current_url}")
                        
                        # Save listing page HTML for debugging selectors
                        listing_html_content = await page.content()
                        listing_safe_filename = f"listing_{sanitize_filename_component(current_url)}.html"
                        with open(os.path.join(html_debug_dir, listing_safe_filename), "w", encoding="utf-8") as f_html:
                            f_html.write(listing_html_content)
                        logger.debug(f"Saved listing page HTML to {listing_safe_filename}")

                        pages_crawled += 1

                        company_links_on_page = await page.locator(listing_selectors["company_link"]).all()
                        logger.debug(f"Found {len(company_links_on_page)} company links on listing page.")
                        
                        detail_page_urls = [await link.get_attribute('href') for link in company_links_on_page]
                        
                        for detail_url in detail_page_urls:
                            if limit > 0 and len(all_discovered_companies_data) >= limit:
                                logger.info(f"Discovery limit of {limit} reached. Stopping.")
                                break

                            if not detail_url: continue
                            
                            full_detail_url = urljoin(current_url, detail_url)
                            normalized_detail_url = normalize_url(full_detail_url)

                            if normalized_detail_url in processed_detail_pages:
                                logger.debug(f"Skipping already processed detail page: {normalized_detail_url}")
                                continue
                            
                            detail_page = await browser.new_page()
                            try:
                                logger.info(f"Navigating to detail page: {normalized_detail_url}")
                                await detail_page.goto(normalized_detail_url, timeout=60000, wait_until='networkidle')
                                
                                # Save HTML for debugging
                                html_content = await detail_page.content()
                                safe_filename = f"{sanitize_filename_component(normalized_detail_url)}.html"
                                with open(os.path.join(html_debug_dir, safe_filename), "w", encoding="utf-8") as f_html:
                                    f_html.write(html_content)
                                logger.debug(f"Saved HTML for {normalized_detail_url} to {safe_filename}")

                                company_data = {}
                                for field, extractor in detail_extractors.items():
                                    element = detail_page.locator(extractor["selector"]).first
                                    data_point = None
                                    if await element.is_visible():
                                        # Log the inner HTML for debugging
                                        inner_html = await element.inner_html()
                                        logger.debug(f"Element for field '{field}' found. Inner HTML: {inner_html[:200]}...")

                                        if extractor["type"] == "text":
                                            data_point = await element.text_content()
                                        elif extractor["type"] == "href":
                                            data_point = await element.get_attribute('href')
                                    else:
                                        logger.debug(f"Element for field '{field}' with selector '{extractor['selector']}' not visible.")
                                        
                                    company_data[field] = data_point.strip() if data_point else None
                                    logger.debug(f"Extracted field '{field}': {company_data[field]}")

                                all_discovered_companies_data.append(company_data)
                                processed_detail_pages.add(normalized_detail_url)

                            except PlaywrightTimeoutError:
                                logger.error(f"Timeout loading detail page: {normalized_detail_url}")
                            except Exception as e_detail:
                                logger.error(f"Error processing detail page {normalized_detail_url}: {e_detail}", exc_info=True)
                            finally:
                                await detail_page.close()

                        if limit > 0 and len(all_discovered_companies_data) >= limit:
                            current_url = None
                        else:
                            next_page_element = page.locator(listing_selectors["pagination_next"]).first
                            if await next_page_element.is_visible():
                                next_page_url = await next_page_element.get_attribute('href')
                                current_url = urljoin(current_url, next_page_url) if next_page_url else None
                                logger.info(f"Navigating to next listing page: {current_url}")
                            else:
                                logger.info("No next page link found. Ending pagination for this start URL.")
                                current_url = None

                    except PlaywrightTimeoutError:
                        logger.error(f"Timeout loading listing page: {current_url}")
                        current_url = None
                    except Exception as e:
                        logger.error(f"An error occurred on listing page {current_url}: {e}", exc_info=True)
                        current_url = None
            
            await browser.close()

    # Run the async main function
    asyncio.run(discovery_main())

    # Save the structured data to a CSV file
    if all_discovered_companies_data:
        output_csv_path = os.path.join(output_dir, f"discovered_data_{profile.get('source_name')}.csv")
        try:
            # The header should be the keys from the detail_page_extractors
            header = list(detail_extractors.keys())
            df = pd.DataFrame(all_discovered_companies_data)
            excel_path = os.path.join(output_dir, f"discovered_data_{profile.get('source_name')}.xlsx")
            df.to_excel(excel_path, index=False)
            logger.info(f"Successfully saved {len(all_discovered_companies_data)} companies to {excel_path}")
        except IOError as e:
            logger.error(f"Failed to write discovered data to CSV file: {e}")

    logger.info(f"Discovery run finished. Found {len(all_discovered_companies_data)} companies.")

def main(args):
    """Main entry point for the discovery pipeline."""
    pipeline_start_time = time.time()
    
    # Initialize AppConfig
    app_config = AppConfig()

    # Setup Logging
    log_file_name = f"discovery_run_{args.profile}_{int(pipeline_start_time)}.log"
    # Note: This assumes a generic output directory structure. This might be refined.
    output_dir = os.path.join(app_config.output_base_dir, f"discovery_{args.profile}_{int(pipeline_start_time)}")
    os.makedirs(output_dir, exist_ok=True)
    
    log_file_path = os.path.join(output_dir, log_file_name)
    file_log_level_int = getattr(logging, app_config.log_level.upper(), logging.INFO)
    console_log_level_str = "DEBUG" if args.debug else app_config.console_log_level.upper()
    console_log_level_int = getattr(logging, console_log_level_str, logging.WARNING)
    setup_logging(file_log_level=file_log_level_int, console_log_level=console_log_level_int, log_file_path=log_file_path)
    
    logger.info(f"Starting Discovery Pipeline for profile: {args.profile}")

    # Load the specified discovery profile
    discovery_profile = load_discovery_profile(args.profile)
    if not discovery_profile:
        logger.error(f"Could not load or find discovery profile: {args.profile}. Exiting.")
        return

    # Run the main pipeline logic
    run_discovery_pipeline(discovery_profile, app_config, output_dir, args.limit)

    total_duration = time.time() - pipeline_start_time
    logger.info(f"Discovery Pipeline for profile '{args.profile}' finished in {total_duration:.2f} seconds.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Discovery Pipeline to find new company URLs from source websites.")
    parser.add_argument(
        "-p", "--profile",
        type=str,
        required=True,
        help="The name of the discovery profile to run (e.g., 'eu_startups')."
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=0,
        help="The maximum number of companies to discover in this run. Default is 0 (no limit)."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug level logging on the console."
    )
    args = parser.parse_args()

    # Basic logging config if no handlers are configured yet
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    main(args)