#!/usr/bin/env python3
"""
Quick script to add gvkey column to existing stock_selected.csv
"""

import pandas as pd
import sys

def add_gvkey_to_stock_selected():
    """Add gvkey column to existing stock selection results"""
    
    print("Adding gvkey to existing stock selection results...")
    
    # Load fundamental data to get tic-to-gvkey mapping
    print("Loading fundamental data for tic-to-gvkey mapping...")
    try:
        fund_data = pd.read_csv("outputs/final_ratios.csv", usecols=['tic', 'gvkey'])
        # Remove duplicates and create mapping
        tic_to_gvkey = fund_data.drop_duplicates(subset=['tic']).set_index('tic')['gvkey'].to_dict()
        print(f"✓ Created mapping for {len(tic_to_gvkey)} unique tic-gvkey pairs")
    except Exception as e:
        print(f"Error loading fundamental data: {str(e)}")
        return False
    
    # Load existing stock selection results
    print("Loading existing stock selection results...")
    try:
        stock_selected = pd.read_csv("results/stock_selected.csv")
        print(f"✓ Loaded {len(stock_selected)} stock selection records")
        print(f"  Columns: {list(stock_selected.columns)}")
    except Exception as e:
        print(f"Error loading stock selection results: {str(e)}")
        return False
    
    # Add gvkey column
    print("Adding gvkey column...")
    stock_selected['gvkey'] = stock_selected['tic'].map(tic_to_gvkey)
    
    # Check for missing gvkeys
    missing_gvkey = stock_selected['gvkey'].isnull().sum()
    if missing_gvkey > 0:
        print(f"⚠️  Warning: {missing_gvkey} records have missing gvkey")
        print("Records with missing gvkey will be dropped")
        stock_selected = stock_selected.dropna(subset=['gvkey'])
    
    # Reorder columns
    stock_selected = stock_selected[['tic', 'gvkey', 'predicted_return', 'trade_date']]
    
    # Save updated results
    output_file = "results/stock_selected.csv"
    stock_selected.to_csv(output_file, index=False)
    print(f"✓ Updated results saved to: {output_file}")
    
    # Display statistics
    print(f"\nFinal statistics:")
    print(f"  Total records: {len(stock_selected)}")
    print(f"  Unique stocks (tic): {stock_selected['tic'].nunique()}")
    print(f"  Unique stocks (gvkey): {stock_selected['gvkey'].nunique()}")
    print(f"  Date range: {stock_selected['trade_date'].min()} to {stock_selected['trade_date'].max()}")
    
    return True

if __name__ == "__main__":
    success = add_gvkey_to_stock_selected()
    if success:
        print("\n✅ Successfully added gvkey to stock selection results!")
    else:
        print("\n❌ Failed to add gvkey to stock selection results!")
        sys.exit(1)
