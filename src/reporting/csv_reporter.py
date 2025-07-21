"""
Handles the generation of CSV reports for prospect analysis.

This module provides functions to take structured prospect analysis data,
flatten it, and write it to a CSV file in a specified output directory.
It includes logic for handling nested data structures and ensuring
consistent output for easier consumption and review.
"""
import os
import logging
from typing import List, Dict, Any, Optional
import pandas as pd

from ..core.schemas import GoldenPartnerMatchOutput, DetailedCompanyAttributes

logger = logging.getLogger(__name__)


def write_sales_outreach_report(
    output_data: List[GoldenPartnerMatchOutput],
    output_dir: str,
    run_id: str,
    original_df: pd.DataFrame,
    sales_prompt_path: Optional[str] = None,
    output_format: str = 'csv'
) -> Optional[str]:
    """
    Writes the sales outreach data to a CSV or Excel file.

    Args:
        output_data (List[GoldenPartnerMatchOutput]): A list of GoldenPartnerMatchOutput objects.
        output_dir (str): The directory where the file will be saved.
        run_id (str): The unique identifier for the current run.
        original_df (pd.DataFrame): The original input DataFrame.
        output_format (str): The output format, either 'csv' or 'excel'.

    Returns:
        Optional[str]: The full path to the saved file, or None if an error occurred.
    """
    if not output_data:
        logger.warning("No output data provided to write_sales_outreach_report. Skipping CSV generation.")
        return None

    try:
        os.makedirs(output_dir, exist_ok=True)
        if output_format == 'excel':
            filename = f"SalesOutreachReport_{run_id}.xlsx"
        else:
            filename = f"SalesOutreachReport_{run_id}.csv"
        full_path = os.path.join(output_dir, filename)

        report_data = []
        

        # Create a mapping from URL to original row for efficient lookup
        original_df_map = {}
        for i in range(len(original_df)):
            row_series = original_df.iloc[i]
            row_dict = row_series.to_dict()
            # Adding 2 because header is 1 and index is 0-based
            row_dict['original_row_number'] = i + 2
            url = row_series.get('GivenURL')
            if url:
                original_df_map[url] = row_dict

        for item in output_data:
            original_url = item.analyzed_company_url
            original_row_data = original_df_map.get(original_url)

            attrs = item.analyzed_company_attributes if item else None
            
            # Initialize a base row structure
            row = {
                'Company Name': None, 'Original_Number': None, 'URL': original_url,
                'is_b2b': None, 'is_b2b_reason': None,
                'serves_1000': None, 'serves_1000_reason': None,
                'found_number': None,
                'sales_pitch': None, 'description': None, 'matched_golden_partner': None,
                'match_reasoning': None,
                'Industry': None,
                'Matched Partner Description': '', 'Avg Leads Per Day': '',
                'Rank': '', 'B2B Indicator': '',
                'Phone Outreach Suitability': '', 'Target Group Size Assessment': '',
                'Products/Services Offered': '', 'USP/Key Selling Points': '',
                'Customer Target Segments': '', 'Business Model': '',
                'Company Size Inferred': '', 'Innovation Level Indicators': '',
                'Website Clarity Notes': '',
                'Original Row Number': None,
                'ScrapeStatus': None
            }

            if original_row_data is not None:
                row.update({
                    'Company Name': original_row_data.get('CompanyName'),
                    'Original_Number': original_row_data.get('Original_Number') or original_row_data.get('PhoneNumber') or original_row_data.get('Number'),
                    'description': original_row_data.get('Description'),
                    'Industry': original_row_data.get('Industry'),
                    'is_b2b': original_row_data.get('is_b2b'),
                    'is_b2b_reason': original_row_data.get('is_b2b_reason'),
                    'serves_1000': original_row_data.get('serves_1000'),
                    'serves_1000_reason': original_row_data.get('serves_1000_reason'),
                    'found_number': original_row_data.get('found_number'),
                    'Original Row Number': original_row_data.get('original_row_number'),
                    'ScrapeStatus': original_row_data.get('ScrapingStatus'),
                })

            if item:
                row.update({
                    'description': item.summary or row.get('Description'),
                    'sales_pitch': item.phone_sales_line.replace('{programmatic placeholder}', f"{item.avg_leads_per_day:.0f}") if item.phone_sales_line and item.avg_leads_per_day is not None else item.phone_sales_line,
                    'match_reasoning': "; ".join(item.match_rationale_features) if item.match_rationale_features else "",
                    'matched_golden_partner': item.matched_partner_name or '',
                    'Matched Partner Description': item.matched_partner_description or '',
                    'Avg Leads Per Day': item.avg_leads_per_day if item.avg_leads_per_day is not None else '',
                    'Rank': item.rank if item.rank is not None else '',
                })

            if attrs:
                row.update({
                    'Industry': attrs.industry or row.get('Industry'),
                    'B2B Indicator': attrs.b2b_indicator or '',
                    'Phone Outreach Suitability': attrs.phone_outreach_suitability or '',
                    'Target Group Size Assessment': attrs.target_group_size_assessment or '',
                    'Products/Services Offered': "; ".join(attrs.products_services_offered) if attrs.products_services_offered else '',
                    'USP/Key Selling Points': "; ".join(attrs.usp_key_selling_points) if attrs.usp_key_selling_points else '',
                    'Customer Target Segments': "; ".join(attrs.customer_target_segments) if attrs.customer_target_segments else '',
                    'Business Model': attrs.business_model or '',
                    'Company Size Inferred': attrs.company_size_category_inferred or '',
                    'Innovation Level Indicators': attrs.innovation_level_indicators_text or '',
                    'Website Clarity Notes': attrs.website_clarity_notes or ''
                })
            
            report_data.append(row)

        if not report_data:
            logger.warning(f"No data to write for sales outreach report. Run ID: {run_id}")
            return None

        df = pd.DataFrame(report_data)
        if output_format == 'excel':
            df.to_excel(full_path, index=False)
            logger.info(f"Successfully wrote sales outreach report to Excel: {full_path}")
        else:
            df.to_csv(full_path, index=False, encoding='utf-8-sig')
            logger.info(f"Successfully wrote sales outreach report to CSV: {full_path}")
        return full_path

    except Exception as e:
        logger.error(f"Error writing sales outreach report to CSV for run_id {run_id}: {e}", exc_info=True)
        return None