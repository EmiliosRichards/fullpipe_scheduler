import asyncio
import json
import os
import logging
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_single_page_extraction():
    """
    Tests the data extraction logic for a single company detail page.
    """
    profile_path = os.path.join(os.path.dirname(__file__), 'discovery_profiles', 'eu_startups.json')
    with open(profile_path, 'r') as f:
        profile = json.load(f)
    
    detail_extractors = profile.get("detail_page_extractors", {})
    test_url = "https://www.eu-startups.com/directory/doc2lang/"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            logger.info(f"Navigating to test URL: {test_url}")
            await page.goto(test_url, timeout=60000, wait_until='networkidle')
            
            logger.info("--- Starting Extraction Test ---")
            
            # Wait for a key element to ensure the page is fully rendered
            try:
                await page.wait_for_selector("h1.wpbdp-listing-title", state="visible", timeout=10000)
                logger.info("Key element 'h1.wpbdp-listing-title' is visible. Proceeding with extraction.")
            except PlaywrightTimeoutError:
                logger.error("Timed out waiting for the key element 'h1.wpbdp-listing-title'. The page may not have loaded correctly.")
                await browser.close()
                return

            extracted_data = {}
            for field, extractor in detail_extractors.items():
                selector = extractor["selector"]
                element = page.locator(selector).first
                data_point = None
                try:
                    if await element.is_visible():
                        if extractor["type"] == "text":
                            data_point = await element.text_content()
                        elif extractor["type"] == "href":
                            data_point = await element.get_attribute('href')
                        logger.info(f"  ✅ SUCCESS: Field '{field}' | Selector '{selector}' | Data: '{data_point}'")
                    else:
                        logger.warning(f"  ❌ FAILED: Field '{field}' | Selector '{selector}' | Reason: Element not visible.")
                except PlaywrightTimeoutError:
                    logger.warning(f"  ❌ FAILED: Field '{field}' | Selector '{selector}' | Reason: Timed out waiting for element.")
                
                extracted_data[field] = data_point.strip() if data_point else None
            
            logger.info("--- Extraction Test Complete ---")
            logger.info("Extracted Data:")
            logger.info(json.dumps(extracted_data, indent=2))

        except Exception as e:
            logger.error(f"An error occurred during the test: {e}", exc_info=True)
        finally:
            await browser.close()

if __name__ == '__main__':
    asyncio.run(test_single_page_extraction())