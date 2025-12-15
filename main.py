import pandas as pd
import numpy as np
import os
import re

from config import *
from utils.cache import get_path
from utils.csv_loader import load_populations, load_landscapes, select_and_load_bids
from utils.model_manager import ModelManager
from utils.processors import (
    parse_location_field, format_location_name, fix_date_format, 
    is_aggregator, check_date_ok, is_gov_related, find_landscape_match
)
from utils.scoring import pop_ok, assign_rank, reasons_for_row

def main():
    print("=== Bid Processor Started ===")

    # 1. Load Resources
    pop_path = get_path('populations.csv')
    print(f"Loading populations from: {pop_path}")
    pops_df = load_populations(pop_path)

    landscape_path = get_path('Landscapes.csv')
    print(f"Loading landscapes from: {landscape_path}")
    landscapes_list = load_landscapes(landscape_path)
    print(f"Loaded {len(landscapes_list)} landscapes.")

    model_mgr = ModelManager()

    # 2. Load Input Data
    raw_df, file_path = select_and_load_bids()
    if raw_df is None:
        return

    print(f"Processing {len(raw_df)} rows from input CSV...")

    # 3. Pre-process Data (Map columns from index)
    try:
        processed_df = pd.DataFrame()
        
        def get_col(idx):
            return raw_df.iloc[:, idx] if idx < len(raw_df.columns) else pd.Series([''] * len(raw_df))

        loc_combined = get_col(COL_IDX_LOCATION_MAIN).astype(str) + ', ' + get_col(COL_IDX_LOCATION_STATE).astype(str)
        
        processed_df['Location'] = loc_combined
        processed_df['Due Date'] = get_col(COL_IDX_DUE_DATE).apply(fix_date_format)
        processed_df['Link'] = get_col(COL_IDX_LINK)
        processed_df['Title'] = get_col(COL_IDX_TITLE)
        processed_df['Phone'] = get_col(COL_IDX_PHONE)

        processed_df = processed_df.dropna(subset=['Location', 'Title', 'Link'], how='all').fillna('')
        processed_df['ID'] = range(1, len(processed_df) + 1)
        
    except Exception as e:
        print(f"Error mapping columns. Check input CSV format. Details: {e}")
        return

    # 4. Location Parsing
    print("Parsing locations...")
    parsed = processed_df['Location'].apply(parse_location_field)
    processed_df['place_name'] = [p[0] for p in parsed]
    processed_df['is_county'] = [p[1] for p in parsed]
    processed_df['state_abbr'] = [p[2] for p in parsed]
    processed_df['discard_flag'] = [p[3] for p in parsed]
    
    processed_df['normalized_location'] = processed_df.apply(
        lambda r: format_location_name(r['place_name'], r['is_county'], r['state_abbr']), 
        axis=1
    )

    # 5. Merge Populations
    print("Merging population data...")
    pops_parsed = pops_df['Location'].astype(str).apply(parse_location_field)
    pops_df['place_name'] = [p[0] for p in pops_parsed]
    pops_df['state_abbr'] = [p[2] for p in pops_parsed]

    merged = processed_df.merge(
        pops_df[['place_name', 'state_abbr', 'Population']],
        how='left',
        on=['place_name', 'state_abbr']
    )

    # 6. Apply Logic Flags
    print("Applying logic flags...")
    merged['is_county_or_city'] = (merged['is_county']) | (merged['place_name'].astype(str).str.lower().str.contains('city'))
    merged['pop_ok'] = merged.apply(pop_ok, axis=1)
    
    merged['is_aggregator'] = merged['Link'].apply(is_aggregator)
    merged['not_aggregator'] = ~merged['is_aggregator']

    merged['is_RFx'] = merged['Title'].astype(str).apply(
        lambda x: bool(re.search(RFX_REGEX, x, flags=re.IGNORECASE))
    )
    merged['not_RFx'] = ~merged['is_RFx']

    merged['due_date_ok'] = check_date_ok(merged['Due Date'])

    merged['Description'] = merged.apply(
        lambda x: f"{x['normalized_location']} is seeking for {x['Title']}", axis=1
    )
    merged['gov_related'] = merged.apply(is_gov_related, axis=1)

    # 7. Model Inference
    print("Running Model Inference...")
    merged['text'] = (merged['Title'].astype(str) + '||' + merged['Description'].astype(str)).str.strip()
    merged['software_prob'] = merged['text'].apply(model_mgr.get_prob)
    merged['software_model_match'] = merged['software_prob'] >= SOFTWARE_THRESHOLD

    # 8. Landscape Matching
    print("Matching Landscapes...")
    merged['Landscapes'] = merged['Title'].apply(lambda t: find_landscape_match(t, landscapes_list))

    # 9. Scoring and Ranking
    print("Calculating scores...")
    weight_flags = ['is_county_or_city', 'pop_ok', 'not_aggregator', 'not_RFx', 'gov_related', 'due_date_ok']
    merged['flags_score'] = merged[weight_flags].sum(axis=1).astype(float)
    merged['pct'] = (merged['flags_score'] / len(weight_flags)) * 100

    merged['Rank'] = merged.apply(assign_rank, axis=1)
    merged['Reasons'] = merged.apply(reasons_for_row, axis=1)

    # 10. Output Generation
    # Added 'Landscapes' to the columns list
    out_cols = ['ID', 'normalized_location', 'Title', 'Link', 'Due Date', 'Rank', 'Landscapes', 'pct', 'software_prob', 'Reasons', 'Description', 'Phone']
    result_df = merged[out_cols].copy()

    # Save
    base_dir = os.path.dirname(file_path)
    output_filename = f"bid_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_path = os.path.join(base_dir, output_filename)
    
    result_df.to_csv(output_path, index=False)
    
    print("\n" + "="*30)
    print(f"SUCCESS! Processed {len(result_df)} rows.")
    print("Rank Distribution:")
    print(result_df['Rank'].value_counts())
    print(f"Results saved to: {output_path}")
    print("="*30)

if __name__ == "__main__":
    main()