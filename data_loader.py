import pandas as pd
from config import logger, DATA_FILES
from typing import Tuple

class MultilingualDataLoader:
    def __init__(self):
        self.data_files = DATA_FILES

    def load_imdb_dataset(self, language: str = "english", n_samples: int = 2000) -> pd.DataFrame:
        filepath = self.data_files.get(language)
        if not filepath:
            logger.error(f"No file path defined for language: {language}")
            return pd.DataFrame()

        logger.info(f"Loading {language} dataset from {filepath}")

        try:
            if language == 'english':
                df = pd.read_csv(filepath)
                df.rename(columns={'review': 'text'}, inplace=True)
            elif language == 'arabic':
                df = pd.read_csv(filepath, sep='\t')
                df.rename(columns={'label': 'sentiment'}, inplace=True)
                df['sentiment'] = df['sentiment'].str.lower()
            else:
                logger.error(f"Loading logic not implemented for {language}")
                return pd.DataFrame()
        except FileNotFoundError:
            logger.error(f"Dataset file not found at: {filepath}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return pd.DataFrame()

        df.dropna(subset=['text', 'sentiment'], inplace=True)
        valid_sentiments = ['positive', 'negative']
        df = df[df['sentiment'].isin(valid_sentiments)]

        n_per_class = n_samples // 2
        pos_df = df[df['sentiment'] == 'positive']
        neg_df = df[df['sentiment'] == 'negative']

        pos_samples = min(len(pos_df), n_per_class)
        neg_samples = min(len(neg_df), n_per_class)

        pos_df = pos_df.sample(n=pos_samples, random_state=42)
        neg_df = neg_df.sample(n=neg_samples, random_state=42)

        final_df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=42).reset_index(drop=True)
        final_df['language'] = language

        logger.info(f"Loaded {len(final_df)} {language} reviews")
        return final_df

    def create_train_test_split(self, df: pd.DataFrame,
                                train_ratio: float = 0.8,
                                dev_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        n = len(df)
        train_end = int(n * train_ratio)
        dev_end = int(n * (train_ratio + dev_ratio))

        train = df[:train_end]
        dev = df[train_end:dev_end]
        test = df[dev_end:]

        logger.info(f"Split sizes - Train: {len(train)}, Dev: {len(dev)}, Test: {len(test)}")
        return train, dev, test