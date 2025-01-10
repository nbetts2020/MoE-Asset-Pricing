# utils/precompute_ebm_data.py

import os
import math
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from utils.utils import get_data, safe_sample, process_group_wrapper
from utils.config import config
import pyarrow as pa
import pyarrow.parquet as pq

def precompute_ebm_data(
    output_path: str,
    percent_data: float = 100.0,
    k: int = 5
):
    """
    Precompute EBM-like data (concatenated articles) for each symbol in the dataset,
    then save to a single Parquet file.

    Args:
        output_path (str): Where to save the final Parquet file (e.g., 'precomputed_ebm.parquet').
        percent_data (float): Percentage of data to load via get_data.
        k (int): Number of preceding articles to include in the concatenation.
    """
    print(f"[Precompute EBM Data] Loading {percent_data}% of the data...")
    df = get_data(percent_data=percent_data)

    # Clean up any weird NaNs, ensure date is sorted
    df = df.dropna(subset=['Date']).reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).reset_index(drop=True)
    df.sort_values('Date', inplace=True)

    # Group by Symbol as in process_data
    grouped = df.groupby('Symbol', sort=False)

    # Prepare arguments
    group_args = [(group_df, k) for _, group_df in grouped]

    # We'll store everything in lists, just like process_data does
    all_articles = []
    all_prices = []
    all_sectors = []
    all_dates = []
    all_related_stocks_list = []
    all_prices_current = []
    all_symbols = []
    all_industries = []
    all_risk_free_rates = []

    # Parallel processing
    num_workers = max(1, cpu_count() - 1)
    print(f"[Precompute EBM Data] Using {num_workers} parallel workers.")
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_group_wrapper, group_args),
            total=len(group_args),
            desc="[Precompute EBM Data] Processing groups"
        ))

    # Flatten
    # results is a list of lists-of-tuples
    # Each tuple => (concatenated_text, price, sector, date, related_stocks, price_current, symbol, industry, risk_free_rate)
    for group in results:
        for item in group:
            (concat_text, price, sector, dt, related, price_cur, symbol, industry, rf_rate) = item
            all_articles.append(concat_text)
            all_prices.append(price)
            all_sectors.append(sector)
            all_dates.append(dt)
            all_related_stocks_list.append(related)
            all_prices_current.append(price_cur)
            all_symbols.append(symbol)
            all_industries.append(industry)
            all_risk_free_rates.append(rf_rate)

    # Build final DataFrame
    out_df = pd.DataFrame({
        'Article': all_articles,
        'weighted_avg_720_hrs': all_prices,
        'Sector': all_sectors,
        'Date': all_dates,
        'RelatedStocksList': all_related_stocks_list,
        'weighted_avg_0_hrs': all_prices_current,
        'Symbol': all_symbols,
        'Industry': all_industries,
        'Risk_Free_Rate': all_risk_free_rates
    })

    # Save to Parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    table = pa.Table.from_pandas(out_df)
    pq.write_table(table, output_path, compression='snappy')
    print(f"[Precompute EBM Data] Done. Wrote {len(out_df)} rows to {output_path}.")


if __name__ == "__main__":
    # Example usage:
    # python -m utils.precompute_ebm_data --output_path=precomputed_ebm.parquet --percent_data=25 --k=5
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default="precomputed_ebm.parquet")
    parser.add_argument('--percent_data', type=float, default=100.0)
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()

    precompute_ebm_data(
        output_path=args.output_path,
        percent_data=args.percent_data,
        k=args.k
    )
