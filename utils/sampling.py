# utils/sampling.py

import pandas as pd
import numpy as np
from typing import List

# Precompute top changes and group by symbol
def prepare_sampling_data(df: pd.DataFrame, k: int, k2: int, k3: int):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    markets_df = df[df['RelatedStocksList'].str.contains("Markets", na=False)]
    grouped_by_symbol = df.groupby('Symbol')

    df['abs_change'] = abs((df['weighted_avg_0_hrs'] - df['weighted_avg_720_hrs']) / df['weighted_avg_720_hrs']) * 100

    top_changes_dict = {
        symbol: group[['Date', 'abs_change']]
        for symbol, group in (
            df.groupby('Symbol')
            .apply(lambda x: x.nlargest(5, 'abs_change'))
        ).groupby(level=0)
    }

    return markets_df, grouped_by_symbol, top_changes_dict

def get_last_articles(stock_df: pd.DataFrame, current_date: pd.Timestamp, n: int = 8) -> str:
    previous_articles = stock_df[stock_df['Date'] < current_date].tail(n)
    formatted_last_articles = [
        f"Previous Article Date: {row['Date'].strftime('%m/%d/%Y')}\n"
        f"Previous Article Title: {row['Title']}\n"
        f"Previous Article Type: {row['articleType']}\n"
        f"Previous Article Publication: {row['Publication']}\n"
        f"Previous Publication Author: {row['Author']}\n"
        f"Previous Article Content: {row['Article']}"
        for _, row in previous_articles.iterrows()
    ]
    return "\n\n".join(formatted_last_articles)

def format_top_changes(stock: str, top_changes_dict: dict) -> str:
    if stock not in top_changes_dict:
        return "No top changes data available for this stock."

    top_changes = top_changes_dict[stock]
    formatted_changes = [
        f"Date: {row['Date'].strftime('%m/%d/%Y')}\nChange: ~{row['abs_change']:.2f}%"
        for _, row in top_changes.iterrows()
    ]
    return "\n\n".join(formatted_changes)

def get_concatenated_articles(row: pd.Series, k: int, k2: int, k3: int, markets_df: pd.DataFrame,
                              grouped_by_symbol: pd.core.groupby.generic.DataFrameGroupBy,
                              top_changes_dict: dict, num_samples: int =1) -> List[str]:
    results = []
    for _ in range(num_samples):
        current_stock = row['Symbol']
        current_industry = row['Industry']
        current_sector = row['Sector']
        article_date = row['Date']

        # Step 1: Sample k "Markets" articles within 30 days before the article date
        eligible_markets = markets_df[
            (markets_df['Date'] <= article_date) &
            (markets_df['Date'] >= article_date - pd.Timedelta(days=30))
        ]
        markets_articles = eligible_markets.sample(n=min(k, len(eligible_markets)), random_state=np.random.randint(0, 10000)).sort_values(by='Date')

        formatted_markets_articles = [
            f"Date: {row['Date'].strftime('%m/%d/%Y')}\nArticle: {row['Article']}"
            for _, row in markets_articles.iterrows()
        ]

        # Step 2: Get k2 articles from the same Industry and k3 from the same Sector within the last 30 days
        industry_articles = df[
            (df['Industry'] == current_industry) &
            (df['Date'] <= article_date) &
            (df['Date'] >= article_date - pd.Timedelta(days=30))
        ]
        formatted_industry_articles = industry_articles.sample(
            n=min(k2, len(industry_articles)), random_state=np.random.randint(0, 10000)
        ).sort_values(by='Date').to_dict('records') if not industry_articles.empty else []

        sector_articles = df[
            (df['Sector'] == current_sector) &
            (df['Date'] <= article_date) &
            (df['Date'] >= article_date - pd.Timedelta(days=30))
        ]
        formatted_sector_articles = sector_articles.sample(
            n=min(k3, len(sector_articles)), random_state=np.random.randint(0, 10000)
        ).sort_values(by='Date').to_dict('records') if not sector_articles.empty else []

        formatted_industry_sector_articles = [
            f"Date: {article['Date'].strftime('%m/%d/%Y')}\nStock: {article['Symbol']}\n"
            f"Industry: {article['Industry']}\nSector: {article['Sector']}\nArticle: {article['Article']}"
            for article in formatted_industry_articles + formatted_sector_articles
        ]

        # Step 3: Get last 8 articles for the current stock directly from grouped data
        stock_df = grouped_by_symbol.get_group(current_stock) if current_stock in grouped_by_symbol.groups else pd.DataFrame()
        last_articles = get_last_articles(stock_df, article_date, n=8)

        # Step 4: Get the precomputed top 5 absolute changes for the current stock
        top_changes = format_top_changes(current_stock, top_changes_dict)

        # Concatenate all parts
        all_articles = "\n\n".join(formatted_markets_articles + formatted_industry_sector_articles + [top_changes, last_articles])
        results.append(all_articles)
    return results
