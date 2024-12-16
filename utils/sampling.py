# utils/sampling.py

import pandas as pd
import numpy as np
import random

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['RelatedStocksList'] = df['RelatedStocksList'].fillna('')
    df['Percentage Change'] = ((df['weighted_avg_0_hrs'] - df['weighted_avg_720_hrs']) / df['weighted_avg_720_hrs']) * 100
    df.sort_values(by='Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def sample_articles(df: pd.DataFrame, index_list):
    # Ensure df['Date'] is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.sort_values(by='Date', inplace=True)
        df.reset_index(drop=True, inplace=True)

    samples = []
    for idx in index_list:
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

        # Define a 30-day time window before the target date
        start_date = target_date - pd.Timedelta(days=30)

        # Filter by date window and ensure articles are before the current article's date
        date_filtered = df[(df['Date'] >= start_date) & (df['Date'] < target_date)]

        def safe_sample(data, n):
            if len(data) == 0:
                return pd.DataFrame()
            return data.sample(n=min(n, len(data)), random_state=random.randint(0, 10000))

        markets_articles = safe_sample(
            date_filtered[date_filtered['RelatedStocksList'].str.contains(r'\bMarkets\b', na=False)],
            5
        )
        industry_articles = safe_sample(
            date_filtered[date_filtered['Industry'] == target_industry],
            5
        )
        sector_articles = safe_sample(
            date_filtered[date_filtered['Sector'] == target_sector],
            5
        )
        stock_articles = df[
            (df['Symbol'] == target_symbol) & (df['Date'] < target_date - pd.Timedelta(days=30))
        ]
        if 'Percentage Change' in stock_articles.columns:
            stock_articles = stock_articles.nlargest(25, 'Percentage Change').head(5)
        else:
            stock_articles = stock_articles.head(5)

        last_8_articles = df[
            (df['Symbol'] == target_symbol) & (df['Date'] < target_date)
        ].sort_values(by='Date', ascending=False).head(8).sort_values(by='Date', ascending=True)

        combined_samples = pd.concat([
            markets_articles, industry_articles, sector_articles,
            stock_articles, last_8_articles
        ]).drop_duplicates()

        current_article = pd.DataFrame([target_row])
        combined_samples = pd.concat([combined_samples, current_article], ignore_index=False)

        combined_samples = combined_samples.head(28)
        samples.append(combined_samples)

    return samples
