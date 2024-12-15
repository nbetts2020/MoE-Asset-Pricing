# utils/data.py

import torch
from torch.utils.data import Dataset
from utils.sampling import preprocess_data, sample_articles  # Ensure these are correctly defined
from utils.config import config
import pandas as pd
import logging
import os
import concurrent.futures

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_concatenated_articles(sample: pd.DataFrame) -> str:
    """
    Formats the concatenated articles from a sample DataFrame.

    Args:
        sample (pd.DataFrame): Sampled DataFrame for a single data point.

    Returns:
        str: Concatenated and formatted article string.
    """
    formatted_articles = []
    idx = sample.iloc[-1].name  # Current index

    # Broader Economic Information (Markets Articles)
    formatted_articles.append("Broader Economic Information:")
    for _, row in sample.iterrows():
        if 'Markets' in row.get('RelatedStocksList', ''):
            date_str = row.get('Date', pd.Timestamp('1970-01-01')).strftime('%Y-%m-%d')
            formatted_articles.append(
                f"Date: {date_str}\n"
                f"Title: {row.get('Title', 'N/A')}\n"
                f"Article: {row.get('Article', 'N/A')}\n"
            )

    # Broader Industry Information
    formatted_articles.append("\nBroader Industry Information:")
    for _, row in sample.iterrows():
        if row.get('Industry', 'Unknown Industry') == sample.iloc[-1].get('Industry', 'Unknown Industry'):
            date_str = row.get('Date', pd.Timestamp('1970-01-01')).strftime('%Y-%m-%d')
            formatted_articles.append(
                f"Date: {date_str}\n"
                f"Title: {row.get('Title', 'N/A')}\n"
                f"Article: {row.get('Article', 'N/A')}\n"
            )

    # Broader Sector Information
    formatted_articles.append("\nBroader Sector Information:")
    for _, row in sample.iterrows():
        if row.get('Sector', 'Unknown Sector') == sample.iloc[-1].get('Sector', 'Unknown Sector'):
            date_str = row.get('Date', pd.Timestamp('1970-01-01')).strftime('%Y-%m-%d')
            formatted_articles.append(
                f"Date: {date_str}\n"
                f"Title: {row.get('Title', 'N/A')}\n"
                f"Article: {row.get('Article', 'N/A')}\n"
            )

    # Information Indicating Significant Market Movement Related to Current Stock
    formatted_articles.append("\nInformation Potentially Indicating Significant Market Movement Related to Current Stock:")
    for _, row in sample.iterrows():
        if (row.get('Symbol', 'Unknown Symbol') == sample.iloc[-1].get('Symbol', 'Unknown Symbol')) and ('Percentage Change' in row):
            date_str = row.get('Date', pd.Timestamp('1970-01-01')).strftime('%Y-%m-%d')
            formatted_articles.append(
                f"Date: {date_str}\n"
                f"Title: {row.get('Title', 'N/A')}\n"
                f"Article: {row.get('Article', 'N/A')}\n"
                f"Percentage Change: {row.get('Percentage Change', 0.0):.2f}%\n"
            )

    # Last 8 Articles for Current Stock
    formatted_articles.append("\nLast 8 Articles for Current Stock:")
    for _, row in sample.iterrows():
        if row.get('Symbol', 'Unknown Symbol') == sample.iloc[-1].get('Symbol', 'Unknown Symbol'):
            date_str = row.get('Date', pd.Timestamp('1970-01-01')).strftime('%Y-%m-%d')
            article_details = (
                f"Symbol: {row.get('Symbol', 'N/A')}\n"
                f"Security: {row.get('Security', 'N/A')}\n"
                f"Related Stocks/Topics: {row.get('RelatedStocksList', 'N/A')}\n"
                f"Title: {row.get('Title', 'N/A')}\n"
                f"Type: {row.get('articleType', 'N/A')}\n"
                f"Publication: {row.get('Publication', 'N/A')}\n"
                f"Publication Author: {row.get('Author', 'N/A')}\n"
                f"Date: {date_str}\n"
                f"Article: {row.get('Article', 'N/A')}\n"
                f"Stock Price 4 days before: {row.get('weighted_avg_-96_hrs', 'N/A')}\n"
                f"Stock Price 2 days before: {row.get('weighted_avg_-48_hrs', 'N/A')}\n"
                f"Stock Price 1 day before: {row.get('weighted_avg_-24_hrs', 'N/A')}\n"
                f"Stock Price at release: {row.get('weighted_avg_0_hrs', 'N/A')}\n"
            )
            formatted_articles.append(article_details)

    concatenated_articles = "\n".join(formatted_articles)
    return concatenated_articles

def worker_generate_samples(args):
    """
    Worker function to generate samples.

    Args:
        args (tuple): Tuple containing (idx, df, num_samples)

    Returns:
        list: List of sample dictionaries.
    """
    idx, df, num_samples = args
    try:
        idx = int(idx)  # Ensure idx is an integer
        logger.info(f"Worker started for idx: {idx}")
        samples = []
        logger.info(f"Num samples to generate: {num_samples} for idx: {idx}")
        for i in range(num_samples):
            logger.debug(f"Generating sample {i+1}/{num_samples} for idx: {idx}")
            sample = sample_articles(df, [idx])[0]  # Pass idx as integer
            concatenated_articles = format_concatenated_articles(sample)

            # Extract scalar values using .iloc[0]
            current_price_series = sample.get('weighted_avg_720_hrs', 0.0)
            current_price = current_price_series.iloc[0] if isinstance(current_price_series, pd.Series) else current_price_series

            current_sector_series = sample.get('Sector', 'Unknown Sector')
            current_sector = current_sector_series.iloc[0] if isinstance(current_sector_series, pd.Series) else current_sector_series

            samples.append({
                'original_idx': idx,  # Store the original index as integer
                'concatenated_articles': concatenated_articles,
                'current_price': current_price,
                'current_sector': current_sector
            })
        logger.info(f"Worker completed for idx: {idx}")
        return samples
    except Exception as e:
        logger.error(f"Error generating samples for idx {idx}: {e}")
        return []

class ArticlePriceDataset(Dataset):
    def __init__(self,
                 articles: list,
                 prices: list,
                 sectors: list,
                 dates: list,
                 related_stocks_list: list,
                 prices_current: list,
                 symbols: list,
                 industries: list,
                 tokenizer,
                 total_epochs: int,
                 use_ebm: bool=False):
        """
        Initializes the dataset with articles, prices, sectors, dates, related_stocks_list, prices_current, symbols, industries, tokenizer, and total_epochs.

        Args:
            articles (list): List of article texts.
            prices (list): List of corresponding prices.
            sectors (list): List of corresponding sectors.
            dates (list): List of corresponding dates.
            related_stocks_list (list): List of related stocks/topics.
            prices_current (list): List of current prices at release.
            symbols (list): List of stock symbols.
            industries (list): List of industries.
            tokenizer: Tokenizer instance for encoding text.
            total_epochs (int): Total number of training epochs.
            use_ebm (bool): Whether to use EBM sampling logic.
        """
        # Create a DataFrame from the provided lists
        self.df = pd.DataFrame({
            'Article': articles,
            'weighted_avg_720_hrs': prices,
            'Sector': sectors,
            'Date': dates,
            'RelatedStocksList': related_stocks_list,
            'weighted_avg_0_hrs': prices_current,
            'Symbol': symbols,
            'Industry': industries
        })
        self.df = preprocess_data(self.df)
        self.tokenizer = tokenizer
        self.total_epochs = total_epochs
        self.use_ebm = use_ebm
        self.sampled_articles = []
        self.current_epoch = 0

    def prepare_epoch(self, current_epoch: int, batch_indices: list):
        """
        Modify the current batch as needed by sampling articles based on batch indices.

        Args:
            current_epoch (int): The current epoch number.
            batch_indices (list): List of indices in the current batch.
        """
        self.current_epoch = current_epoch
        if self.use_ebm:
            num_samples = max(self.total_epochs - current_epoch, 5)
            if num_samples < 1:
                num_samples = 1
            # Convert batch_indices to integers and ensure they are within bounds
            batch_indices = [int(idx) for idx in batch_indices if int(idx) < len(self.df)]

            self.sampled_articles = []

            # Prepare arguments for worker_generate_samples
            args_list = [(idx, self.df, num_samples) for idx in batch_indices]

            # Parallel processing using ProcessPoolExecutor
            num_workers = os.cpu_count() or 1
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                    results = executor.map(worker_generate_samples, args_list)
                    for sample_batch in results:
                        self.sampled_articles.extend(sample_batch)
                logger.info(f"Epoch {current_epoch}: Prepared {len(self.sampled_articles)} sampled articles for the current batch.")
            except Exception as e:
                logger.error(f"Error during parallel sampling: {e}")
                self.sampled_articles = []
        else:
            # Do nothing
            pass

    def generate_context_string(self, original_idx):
        """
        Generates a concatenated context string for a given original index.

        Args:
            original_idx (int): Original index of the data point in the DataFrame.

        Returns:
            str: Concatenated context string.
        """
        sample = self.df.iloc[original_idx]
        target_date = sample.get('Date', pd.Timestamp('1970-01-01'))
        target_symbol = sample.get('Symbol', 'Unknown Symbol')
        target_sector = sample.get('Sector', 'Unknown Sector')
        target_industry = sample.get('Industry', 'Unknown Industry')

        # Define a 30-day time window before the target date
        start_date = target_date - pd.Timedelta(days=30)

        # Filter articles within the date range and before the target date
        date_filtered = self.df[
            (self.df['Date'] >= start_date) &
            (self.df['Date'] < target_date)
        ]

        # Helper function to sample articles
        def sample_articles_subset(dataframe, n_samples):
            if len(dataframe) >= n_samples:
                return dataframe.sample(n_samples, random_state=42)
            else:
                return dataframe

        # Generate contexts from different categories
        contexts = []

        # Broader Economic Information
        economic_articles = date_filtered[
            date_filtered['RelatedStocksList'].str.contains(r'\bMarkets\b', na=False)
        ]
        economic_contexts = sample_articles_subset(economic_articles, 2)['Article'].tolist()

        # Industry-Specific Information
        industry_articles = date_filtered[
            date_filtered['Industry'] == target_industry
        ]
        industry_contexts = sample_articles_subset(industry_articles, 2)['Article'].tolist()

        # Sector-Specific Information
        sector_articles = date_filtered[
            date_filtered['Sector'] == target_sector
        ]
        sector_contexts = sample_articles_subset(sector_articles, 2)['Article'].tolist()

        # Stock-Specific Information (Top movers)
        stock_articles = self.df[
            (self.df['Symbol'] == target_symbol) &
            (self.df['Date'] < target_date - pd.Timedelta(days=30))
        ]

        if 'Percentage Change' in stock_articles.columns:
            stock_articles = stock_articles.nlargest(25, 'Percentage Change')
            stock_contexts = sample_articles_subset(stock_articles, 2)['Article'].tolist()
        else:
            logger.warning("'Percentage Change' column missing. Skipping stock-specific information.")
            stock_contexts = []

        # Last 8 Articles
        last_8_articles = self.df[
            (self.df['Symbol'] == target_symbol) &
            (self.df['Date'] < target_date)
        ].sort_values(by='Date', ascending=False).head(8)
        last_8_contexts = last_8_articles['Article'].tolist()

        # Combine contexts into a single string
        contexts.extend(economic_contexts)
        contexts.extend(industry_contexts)
        contexts.extend(sector_contexts)
        contexts.extend(stock_contexts)
        contexts.extend(last_8_contexts)

        # Ensure all contexts are strings
        contexts = [str(context) for context in contexts]

        concatenated_contexts = " ".join(contexts)
        return concatenated_contexts

    def __len__(self):
        return len(self.sampled_articles) if self.use_ebm and self.sampled_articles else len(self.df)

    def __getitem__(self, idx):
        if self.use_ebm and self.sampled_articles:
            try:
                sample = self.sampled_articles[idx]
                encoding = self.tokenizer(
                    sample['concatenated_articles'],
                    truncation=True,
                    padding='max_length',
                    max_length=config.BLOCK_SIZE,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].squeeze(0)                     # Shape: (seq_len,)
                label = torch.tensor(sample['current_price'], dtype=torch.float) # Scalar
                sector = sample['current_sector']                                # String

                # Use the original DataFrame index for context generation
                original_idx = sample['original_idx']
                concatenated_contexts = self.generate_context_string(original_idx)

                # Tokenize concatenated contexts
                context_encoding = self.tokenizer(
                    concatenated_contexts,
                    truncation=True,
                    padding='max_length',
                    max_length=config.BLOCK_SIZE,
                    return_tensors='pt'
                )
                context_input_ids = context_encoding['input_ids'].squeeze(0)     # Shape: (seq_len,)

                return {
                    'input_ids': input_ids,                     # Shape: (seq_len,)
                    'labels': label,                            # Scalar
                    'sector': sector,                           # String
                    'context_input_ids': context_input_ids,     # Shape: (seq_len,)
                    'idx': original_idx                         # Include original index as integer
                }
            except IndexError:
                logger.error(f"Index {idx} out of range for sampled_articles with length {len(self.sampled_articles)}.")
                raise
            except Exception as e:
                logger.error(f"Error in __getitem__ for idx {idx}: {e}")
                raise
        else:
            # Fallback to original behavior if sampling not used
            try:
                row = self.df.iloc[idx]
                article = row.get('Article', 'N/A')
                price = row.get('weighted_avg_720_hrs', 0.0)
                sector = row.get('Sector', 'Unknown Sector')
                encoding = self.tokenizer(
                    article,
                    truncation=True,
                    padding='max_length',
                    max_length=config.BLOCK_SIZE,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].squeeze(0)
                return {
                    'input_ids': input_ids,
                    'labels': torch.tensor(price, dtype=torch.float),
                    'sector': sector,
                    'idx': int(idx)
                }
            except IndexError:
                logger.error(f"Index {idx} out of range for DataFrame with length {len(self.df)}.")
                raise
            except Exception as e:
                logger.error(f"Error in __getitem__ for idx {idx}: {e}")
                raise

def custom_collate_fn(batch):
    """
    Custom collate function to handle batch indices.

    Args:
        batch (list): List of samples returned by __getitem__.

    Returns:
        dict: Batched inputs.
    """
    try:
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        sectors = [item['sector'] for item in batch]
        idxs = [item['idx'] for item in batch]  # Already integers

        # Handle context_input_ids if present
        if 'context_input_ids' in batch[0]:
            context_input_ids = torch.stack([item['context_input_ids'] for item in batch])
            return {
                'input_ids': input_ids,
                'labels': labels,
                'sector': sectors,
                'context_input_ids': context_input_ids,
                'idx': idxs
            }
        else:
            return {
                'input_ids': input_ids,
                'labels': labels,
                'sector': sectors,
                'idx': idxs
            }
    except KeyError as e:
        logger.error(f"Missing key in batch during collate: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in custom_collate_fn: {e}")
        raise
