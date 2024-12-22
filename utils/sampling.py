# utils/sampling.py

import pandas as pd
import numpy as np
import random

def sample_articles(df: pd.DataFrame, index_list=None, symbol=None):

    samples = []
    if index_list is not None:
        for idx in index_list:
            if idx >= len(df) or idx < 0:
                logging.error(f"Index {idx} is out of bounds. Skipping.")
                continue
    
            target_row = df.loc[idx]
            target_date = target_row['Date']
            if pd.isna(target_date):
                logging.error(f"Index {idx} has invalid Date. Skipping.")
                continue
            if not isinstance(target_date, pd.Timestamp):
                # Attempt conversion if still string for some reason
                target_date = pd.to_datetime(target_date, errors='coerce')
                if pd.isna(target_date):
                    logging.error(f"Could not convert target_date for idx {idx}. Skipping.")
                    continue

            target_symbol = target_row['Symbol']
            target_sector = target_row['Sector']
            target_industry = target_row['Industry']
    else:
        target_symbol = target_row['Symbol']
        target_sector = (
            df[df['Symbol'] == target_symbol]['Sector']
            .value_counts()
            .idxmax()
        )
        target_industry = (
            df[df['Symbol'] == target_symbol]['Industry']
            .value_counts()
            .idxmax()
        )

        # Define a 30-day time window before the target date
        start_date = target_date - pd.Timedelta(days=30)

        # Filter by date window and ensure articles are before the current article's date
        date_filtered = df[(df['Date'] >= start_date) & (df['Date'] < target_date)]

        def safe_sample(data, n):
            if len(data) == 0:
                return pd.DataFrame()
            return data.sample(n=min(n, len(data)), random_state=random.randint(0, 10000))

        markets_articles = safe_sample(
            date_filtered[(date_filtered['RelatedStocksList'].str.contains(r'\bMarkets\b', na=False)) & (date_filtered['Symbol'] != target_symbol)],
            5
        )
        industry_articles = safe_sample(
            date_filtered[(date_filtered['Industry'] == target_industry) & (date_filtered['Symbol'] != target_symbol)],
            5
        )
        sector_articles = safe_sample(
            date_filtered[(date_filtered['Sector'] == target_sector) & (date_filtered['Symbol'] != target_symbol)],
            5
        )
        stock_articles = df[
            (df['Symbol'] == target_symbol) & (df['Date'] < target_date - pd.Timedelta(days=30))
        ]
        stock_articles = stock_articles.dropna(subset=['Percentage Change'])
        stock_articles = stock_articles.nlargest(25, 'Percentage Change').head(5)

        last_8_articles = df[
            (df['Symbol'] == target_symbol) & (df['Date'] < target_date)
        ].sort_values(by='Date', ascending=False).head(8).sort_values(by='Date', ascending=True)

        # Create a dictionary for each category
        sample_dict = {
            'markets': markets_articles,
            'industry': industry_articles,
            'sector': sector_articles,
            'stock': stock_articles,
            'last_8': last_8_articles,
            'current': pd.DataFrame([target_row])
        }

        samples.append(sample_dict)

    return samples
