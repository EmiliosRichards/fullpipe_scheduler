import logging
import re
import hashlib
from urllib.parse import urljoin, urlparse, urldefrag
from bs4 import BeautifulSoup
from bs4.element import Tag
from typing import List, Tuple, Optional, Any
import httpx

from ..core.config import AppConfig

config_instance = AppConfig()
logger = logging.getLogger(__name__)

def normalize_url(url: str) -> str:
    """
    Normalizes a URL to a canonical form.
    """
    try:
        url_no_frag, _ = urldefrag(url)
        parsed = urlparse(url_no_frag)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parsed.path
        common_indexes = ['index.html', 'index.htm', 'index.php', 'default.html', 'default.htm', 'index.asp', 'default.asp']
        for index_file in common_indexes:
            if path.endswith(f'/{index_file}'):
                path = path[:-len(index_file)]
                break
        if netloc and path and not path.startswith('/'):
            path = '/' + path
        if path != '/' and path.endswith('/'):
            path = path[:-1]
        if not path and netloc:
            path = '/'
        query = ''
        if parsed.query:
            params = parsed.query.split('&')
            ignored_params = {'fallback'}
            filtered_params = [p for p in params if (p.split('=')[0].lower() if '=' in p else p.lower()) not in ignored_params]
            if filtered_params:
                query = '&'.join(sorted(filtered_params))
        return urlparse('')._replace(scheme=scheme, netloc=netloc, path=path, params=parsed.params, query=query, fragment='').geturl()
    except Exception as e:
        logger.error(f"Error normalizing URL '{url}': {e}. Returning original URL.", exc_info=True)
        return url

def get_safe_filename(name_or_url: str, for_url: bool = False, max_len: int = 100) -> str:
    if for_url:
        logger.info(f"get_safe_filename (for_url=True): Input for filename generation='{name_or_url}'")
    original_input = name_or_url
    if for_url:
        parsed_original_url = urlparse(original_input)
        domain_part = re.sub(r'^www\.', '', parsed_original_url.netloc)
        domain_part = re.sub(r'[^\w-]', '', domain_part)[:config_instance.filename_url_domain_max_len]
        url_hash = hashlib.sha256(original_input.encode('utf-8')).hexdigest()[:config_instance.filename_url_hash_max_len]
        safe_name = f"{domain_part}_{url_hash}" # Use the sanitized domain_part
        logger.info(f"DEBUG PATH: get_safe_filename (for_url=True) output: '{safe_name}' from input '{original_input}'") # DEBUG PATH LENGTH
        return safe_name
    else:
        name_or_url = re.sub(r'^https?://', '', name_or_url)
        safe_name = re.sub(r'[^\w.-]', '_', name_or_url)
        safe_name_truncated = safe_name[:max_len]
        logger.info(f"DEBUG PATH: get_safe_filename (for_url=False) output: '{safe_name_truncated}' (original sanitized: '{safe_name}', max_len: {max_len}) from input '{original_input}'") # DEBUG PATH LENGTH
        return safe_name_truncated

def extract_text_from_html(html_content: str) -> str:
    if not html_content: return ""
    soup = BeautifulSoup(html_content, 'html.parser')
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    text = soup.get_text(separator=' ', strip=True)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def find_links(
    html_content: str,
    base_url: str,
    input_row_id: Any,
    company_name_or_id: str,
    extract_external: bool = False,
    company_link_selector: Optional[str] = None
) -> List[Tuple[str, int]]:
    if not html_content: return []
    
    scored_links: List[Tuple[str, int]] = []
    soup = BeautifulSoup(html_content, 'html.parser')
    normalized_base_url_str = normalize_url(base_url)
    parsed_base_url = urlparse(normalized_base_url_str)

    link_tags = soup.select(company_link_selector) if company_link_selector else soup.find_all('a', href=True)

    for link_tag in link_tags:
        if not isinstance(link_tag, Tag): continue
        href_attr = link_tag.get('href')
        current_href: Optional[str] = None
        if isinstance(href_attr, str): current_href = href_attr.strip()
        elif isinstance(href_attr, list) and href_attr and isinstance(href_attr[0], str): current_href = href_attr[0].strip()
        if not current_href: continue

        absolute_url_raw = urljoin(base_url, current_href)
        normalized_link_url = normalize_url(absolute_url_raw)
        parsed_normalized_link = urlparse(normalized_link_url)

        if parsed_normalized_link.scheme not in ['http', 'https']: continue
        
        is_external = parsed_normalized_link.netloc != parsed_base_url.netloc
        
        if extract_external:
            if is_external:
                # Simple scoring for external links, can be enhanced later
                score = 100
                scored_links.append((normalized_link_url, score))
            continue # Skip internal links when in external mode

        # --- Internal Link Logic (existing logic) ---
        if is_external: continue # Skip external links when in internal mode

        link_text = link_tag.get_text().lower().strip()
        link_href_lower = normalized_link_url.lower()
        initial_keyword_match = False
        if config_instance.target_link_keywords:
            if any(kw in link_text for kw in config_instance.target_link_keywords) or \
               any(kw in link_href_lower for kw in config_instance.target_link_keywords):
                initial_keyword_match = True
        if not initial_keyword_match: continue

        if config_instance.scraper_exclude_link_path_patterns:
            path_lower = parsed_normalized_link.path.lower()
            if any(p and p in path_lower for p in config_instance.scraper_exclude_link_path_patterns):
                logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Link '{normalized_link_url}' hard excluded by pattern in path: '{path_lower}'.")
                continue
        
        score = 0
        path_segments = [seg for seg in parsed_normalized_link.path.lower().strip('/').split('/') if seg]
        num_segments = len(path_segments)

        if config_instance.scraper_critical_priority_keywords:
            for crit_kw in config_instance.scraper_critical_priority_keywords:
                if any(seg == crit_kw for seg in path_segments):
                    score = max(score, 100 - min(20, (num_segments - config_instance.scraper_max_keyword_path_segments) * 5) if num_segments > config_instance.scraper_max_keyword_path_segments else 100)
                    if score >= 100: break
        
        if score < 90 and config_instance.scraper_high_priority_keywords:
            for high_kw in config_instance.scraper_high_priority_keywords:
                if any(seg == high_kw for seg in path_segments):
                    score = max(score, 90 - min(20, (num_segments - config_instance.scraper_max_keyword_path_segments) * 5) if num_segments > config_instance.scraper_max_keyword_path_segments else 90)
                    if score >= 90: break

        if score >= config_instance.scraper_min_score_to_queue:
            log_text_snippet = link_text[:50].replace('\n', ' ')
            logger.debug(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Link '{normalized_link_url}' scored: {score} (Text: '{log_text_snippet}...', Path: '{parsed_normalized_link.path}') - Adding to potential queue.")
            scored_links.append((normalized_link_url, score))
            
    logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] From page {base_url}, found {len(scored_links)} links meeting criteria (External={extract_external}).")
    return scored_links

def _classify_page_type(url_str: str, config: AppConfig) -> str:
    """Classifies a URL based on keywords in its path."""
    if not url_str:
        return "unknown"
    
    url_lower = url_str.lower()
    # Check for specific page types based on keywords in URL path
    # Order matters if keywords overlap; more specific should come first if necessary.
    # For now, assuming simple first-match.
    
    # Path-based classification
    parsed_url = urlparse(url_lower)
    path_lower = parsed_url.path

    # New page type classification
    if hasattr(config, 'page_type_keywords_about') and any(kw in path_lower for kw in config.page_type_keywords_about):
        return "about"
    if hasattr(config, 'page_type_keywords_product_service') and any(kw in path_lower for kw in config.page_type_keywords_product_service):
        return "product_service"
    # Add other new classifications here if needed, e.g.:
    # if hasattr(config, 'page_type_keywords_blog') and any(kw in path_lower for kw in config.page_type_keywords_blog):
    #     return "blog"

    # Fallback if no path keywords match, check full URL for very generic terms
    # (less reliable, path is usually better indicator for specific types)
    if hasattr(config, 'page_type_keywords_about') and any(kw in url_lower for kw in config.page_type_keywords_about):
        return "about"
    if hasattr(config, 'page_type_keywords_product_service') and any(kw in url_lower for kw in config.page_type_keywords_product_service):
        return "product_service"
    # Add other new classifications here for full URL check if needed

    # If it's just the base domain (e.g., http://example.com or http://example.com/)
    if not path_lower or path_lower == '/':
        return "homepage"

    return "general_content"

async def validate_link_status(url: str, http_client: httpx.AsyncClient) -> bool:
    """
    Validates the status of a URL by performing a HEAD request.

    Args:
        url: The URL to validate.
        http_client: An httpx.AsyncClient instance.

    Returns:
        True if the URL is valid (2xx status code), False otherwise.
    """
    try:
        response = await http_client.head(url, timeout=10, follow_redirects=True)
        if 200 <= response.status_code < 300:
            return True
        else:
            logger.warning(f"Skipping broken link (status {response.status_code}): {url}")
            return False
    except httpx.RequestError as e:
        logger.warning(f"Skipping link due to request error: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during link validation for {url}: {e}", exc_info=True)
        return False