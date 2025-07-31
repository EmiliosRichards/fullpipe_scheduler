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
    request_delay = profile.get("request_delay_seconds", 5) # Default to 5 seconds

    if not all([start_urls, listing_selectors, detail_extractors]):
        logger.error("Profile is missing one or more required keys: 'start_urls', 'listing_page_selectors', 'detail_page_extractors'.")
        return

    processed_detail_pages = set()
    discovered_companies_count = 0

    # Prepare output file
    excel_path = os.path.join(output_dir, f"discovered_data_{profile.get('source_name')}.xlsx")
    header = list(detail_extractors.keys())
    # Create an empty file with header to start
    pd.DataFrame(columns=header).to_excel(excel_path, index=False, engine='openpyxl')

    async def discovery_main():
        nonlocal discovered_companies_count
        # Create a subdirectory for saving HTML content for debugging
        html_debug_dir = os.path.join(output_dir, "html_debug")
        os.makedirs(html_debug_dir, exist_ok=True)
        
        async with async_playwright() as p:
            launch_options = {"headless": True}
            if app_config.proxy_server:
                launch_options["proxy"] = {
                    "server": app_config.proxy_server
                }
                logger.info(f"Using proxy server: {app_config.proxy_server}")
            browser = await p.chromium.launch(**launch_options)
            page = await browser.new_page()

            for start_url in start_urls:
                if limit > 0 and discovered_companies_count >= limit: break
                
                logger.info(f"Processing start URL: {start_url}")
                current_url: Optional[str] = start_url
                pages_crawled = 0
                
                # Check for progress file and resume if it exists
                progress_file = os.path.join(output_dir, f"progress_{sanitize_filename_component(start_url)}.txt")
                if os.path.exists(progress_file):
                    with open(progress_file, 'r') as f:
                        resume_url = f.read().strip()
                        if resume_url:
                            logger.info(f"Resuming from saved progress: {resume_url}")
                            current_url = resume_url

                while current_url:
                    # If max_pages is set (non-zero), stop if the limit is reached.
                    if max_pages > 0 and pages_crawled >= max_pages:
                        logger.info(f"Reached max_pages limit of {max_pages}. Stopping pagination for this start URL.")
                        break
                    if limit > 0 and discovered_companies_count >= limit: break
                    
                    try:
                        await page.goto(current_url, timeout=60000, wait_until='networkidle')
                        logger.info(f"Successfully navigated to listing page: {current_url}")
                        await asyncio.sleep(request_delay) # Add delay
                        
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
                            if limit > 0 and discovered_companies_count >= limit:
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
                                await asyncio.sleep(request_delay) # Add delay
                                
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

                                # Append data incrementally
                                df_single = pd.DataFrame([company_data], columns=header)
                                with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                                    df_single.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)

                                discovered_companies_count += 1
                                processed_detail_pages.add(normalized_detail_url)

                            except PlaywrightTimeoutError:
                                logger.error(f"Timeout loading detail page: {normalized_detail_url}")
                            except Exception as e_detail:
                                logger.error(f"Error processing detail page {normalized_detail_url}: {e_detail}", exc_info=True)
                            finally:
                                await detail_page.close()

                        if limit > 0 and discovered_companies_count >= limit:
                            current_url = None
                        else:
                            next_page_element = page.locator(listing_selectors["pagination_next"]).first
                            if await next_page_element.is_visible():
                                next_page_url = await next_page_element.get_attribute('href')
                                current_url = urljoin(current_url, next_page_url) if next_page_url else None
                                if current_url:
                                    # Save the next page URL as progress
                                    with open(progress_file, 'w') as f:
                                        f.write(current_url)
                                    logger.info(f"Navigating to next listing page: {current_url}")
                                else:
                                    logger.info("No next page link found. Ending pagination for this start URL.")
                                    # Clear progress file on completion
                                    if os.path.exists(progress_file):
                                        os.remove(progress_file)
                            else:
                                logger.info("No next page link found. Ending pagination for this start URL.")
                                current_url = None
                                # Clear progress file on completion
                                if os.path.exists(progress_file):
                                    os.remove(progress_file)

                    except PlaywrightTimeoutError:
                        logger.error(f"Timeout loading listing page: {current_url}")
                        current_url = None
                    except Exception as e:
                        logger.error(f"An error occurred on listing page {current_url}: {e}", exc_info=True)
                        current_url = None
            
            await browser.close()

    # Run the async main function
    asyncio.run(discovery_main())

    logger.info(f"Discovery run finished. Found {discovered_companies_count} companies.")

def main(args):
    """Main entry point for the discovery pipeline."""
    pipeline_start_time = time.time()
    
    # Initialize AppConfig and update with CLI args
    app_config = AppConfig()
    if args.proxy_server:
        app_config.proxy_server = args.proxy_server

    # Setup base logging for the overall run
    overall_log_dir = os.path.join(app_config.output_base_dir, f"discovery_run_{int(pipeline_start_time)}")
    os.makedirs(overall_log_dir, exist_ok=True)
    
    log_file_path = os.path.join(overall_log_dir, f"main_run_{int(pipeline_start_time)}.log")
    file_log_level_int = getattr(logging, app_config.log_level.upper(), logging.INFO)
    console_log_level_str = "DEBUG" if args.debug else app_config.console_log_level.upper()
    console_log_level_int = getattr(logging, console_log_level_str, logging.WARNING)
    setup_logging(file_log_level=file_log_level_int, console_log_level=console_log_level_int, log_file_path=log_file_path)

    logger.info(f"Starting Discovery Pipeline for profiles: {', '.join(args.profiles)}")

    for profile_name in args.profiles:
        profile_start_time = time.time()
        logger.info(f"--- Processing profile: {profile_name} ---")

        # Load the specified discovery profile
        discovery_profile = load_discovery_profile(profile_name)
        if not discovery_profile:
            logger.error(f"Could not load or find discovery profile: {profile_name}. Skipping.")
            continue

        # Create a dedicated output directory for this profile
        profile_output_dir = os.path.join(overall_log_dir, f"discovery_{profile_name}_{int(profile_start_time)}")
        os.makedirs(profile_output_dir, exist_ok=True)
        
        # You might want to set up profile-specific logging here if needed,
        # but for now, we'll use the main logger.

        # Run the main pipeline logic
        run_discovery_pipeline(discovery_profile, app_config, profile_output_dir, args.limit)
        
        profile_duration = time.time() - profile_start_time
        logger.info(f"--- Finished processing profile '{profile_name}' in {profile_duration:.2f} seconds. ---")

    total_duration = time.time() - pipeline_start_time
    logger.info(f"All discovery profiles processed. Total run time: {total_duration:.2f} seconds.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Discovery Pipeline to find new company URLs from source websites.")
    parser.add_argument(
        "-p", "--profiles",
        nargs='+',
        type=str,
        required=True,
        help="One or more discovery profile names to run (e.g., 'de_startups' 'at_startups')."
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
    parser.add_argument(
        "--proxy-server",
        type=str,
        default=None,
        help="Proxy server to use for requests (e.g., 'http://user:pass@host:port')."
    )
    args = parser.parse_args()

    # Basic logging config if no handlers are configured yet
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    main(args)