import logging
from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError
from typing import List, Dict, Any
from ..core.config import AppConfig

logger = logging.getLogger(__name__)

class InteractionHandler:
    def __init__(self, page: Page, config: AppConfig):
        self.page = page
        self.config = config

    async def handle_interactions(self) -> bool:
        """
        Performs a limited number of quick passes to close modal dialogs,
        cookie banners, etc., based on configured selectors and text queries.
        Returns True if any interaction was successfully handled.
        """
        if not self.config.interaction_handler_enabled:
            logger.debug("Interaction handler is disabled in the configuration.")
            return False

        max_passes = self.config.interaction_handler_max_passes
        visibility_timeout = self.config.interaction_handler_visibility_timeout_ms
        any_interaction_handled = False

        for i in range(max_passes):
            handled_in_pass = False
            interactions = [("selector", s) for s in self.config.interaction_selectors] + \
                           [("text", t) for t in self.config.interaction_text_queries]

            for type, query in interactions:
                try:
                    element = None
                    if type == "selector":
                        element = self.page.locator(query).first
                    else: # text
                        element = self.page.locator(f"*:visible:text-matches('{query}', 'i')").first
                    
                    if await element.is_visible():
                        logger.info(f"Found and clicking element by {type}: '{query}'")
                        await element.click(timeout=1000)
                        handled_in_pass = True
                        any_interaction_handled = True
                        await self.page.wait_for_timeout(500) # wait for UI to settle
                        break # break from the for loop to restart the pass
                except PlaywrightTimeoutError:
                    # This is expected if the element is not visible within the short timeout
                    logger.debug(f"Element not visible or timed out for {type} '{query}'.")
                except Exception as e:
                    logger.warning(f"Error handling {type} '{query}': {e}")
            
            if not handled_in_pass:
                # If a full pass completes with no interactions handled, we can exit early.
                logger.debug(f"No interactive elements found in pass {i+1}/{max_passes}. Exiting handler.")
                return any_interaction_handled
        
        logger.debug(f"Completed {max_passes} interaction handling passes.")
        return any_interaction_handled

    async def execute_navigation_steps(self, navigation_steps: List[Dict[str, Any]]):
        """
        Executes a predefined series of navigation steps from a discovery profile.

        Args:
            navigation_steps (List[Dict[str, Any]]): A list of action dictionaries.
        """
        if not navigation_steps:
            return

        logger.info(f"Executing {len(navigation_steps)} custom navigation steps.")
        for i, step in enumerate(navigation_steps):
            action = step.get("action")
            selector = step.get("selector")
            timeout = step.get("timeout", self.config.default_page_timeout)
            logger.info(f"Executing Step {i+1}: action='{action}', selector='{selector}', timeout={timeout}")

            try:
                if action == "wait_for_selector":
                    if selector:
                        await self.page.wait_for_selector(selector, state='visible', timeout=timeout)
                        logger.info(f"Successfully waited for selector: {selector}")
                elif action == "click":
                    if selector:
                        await self.page.click(selector, timeout=timeout)
                        logger.info(f"Successfully clicked selector: {selector}")
                elif action == "scroll_to_bottom":
                    await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    logger.info("Successfully scrolled to bottom of page.")
                # Add other actions like 'delay' or 'fill_input' here in the future
                
                # Optional delay after an action
                delay = step.get("delay_after_ms", 500)
                await self.page.wait_for_timeout(delay)

            except PlaywrightTimeoutError:
                logger.error(f"Timeout error during navigation step {i+1} ('{action}') for selector '{selector}' after {timeout}ms.")
                # Decide if the pipeline should stop or continue on error
                raise  # Reraise to let the caller handle it
            except Exception as e:
                logger.error(f"An error occurred during navigation step {i+1} ('{action}'): {e}", exc_info=True)
                raise