import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_FILES = {
    'english': 'IMDB Dataset.csv',
    'arabic': 'ar_reviews_100k.tsv'
}

OUTPUT_FILE = 'pipeline_results.json'

TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
SAMPLE_SIZE = 1200 