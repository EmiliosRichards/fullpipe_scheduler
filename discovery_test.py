import os
import subprocess
import sys
import pandas as pd
import pytest

def test_discovery_pipeline():
    """
    Runs an integration test of the discovery pipeline.
    """
    profile_name = "eu_startups"
    
    # --- Execution ---
    print(f"--- Running Discovery Pipeline for profile: {profile_name} ---")
    process = subprocess.Popen(
        [sys.executable, "discovery_pipeline.py", "--profile", profile_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()

    print("--- Pipeline Output ---")
    print(stdout.decode('latin-1', errors='ignore'))
    if stderr:
        print("--- Errors ---")
        print(stderr.decode('latin-1', errors='ignore'))
    print("--- End Pipeline Output ---")

    assert process.returncode == 0, "Discovery pipeline script exited with an error."

    # --- Validation ---
    print("--- Validating test results ---")
    
    # Find the latest discovery output directory
    output_dirs = [d for d in os.listdir("output_data") if d.startswith(f"discovery_{profile_name}")]
    assert output_dirs, "No output directory found for the discovery run."
    latest_run_dir = max(output_dirs, key=lambda d: os.path.getmtime(os.path.join("output_data", d)))
    output_dir_path = os.path.join("output_data", latest_run_dir)
    
    output_csv_path = os.path.join(output_dir_path, f"discovered_data_{profile_name}.csv")
    
    assert os.path.exists(output_csv_path), f"Output CSV file not found at: {output_csv_path}"
    
    # Check that the CSV contains data
    result_df = pd.read_csv(output_csv_path)
    assert not result_df.empty, "The output CSV file is empty."
    
    # Check for expected columns based on the profile
    expected_columns = [
        "company_name", "website", "founded", "tags", "based_in", 
        "business_description", "long_business_description", "category"
    ]
    for col in expected_columns:
        assert col in result_df.columns, f"Expected column '{col}' not found in the output CSV."

    print("\n🎉 Discovery pipeline test passed! 🎉")
