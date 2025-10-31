import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# ============================================================================
# Import Libraries
# ============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import logging
import re
import string
from collections import defaultdict, Counter
import pickle

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag as nltk_pos_tag
from nltk import RegexpParser, Tree

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from xgboost import XGBClassifier

import pyarabic.araby as araby
from camel_tools.disambig.mle import MLEDisambiguator

import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Data Loader Class
# ============================================================================

class MultilingualDataLoader:
    """Load and manage multilingual movie review datasets from real files."""

    def __init__(self):
        self.data_files = {
            'english': 'IMDB Dataset.csv',
            'arabic': 'ar_reviews_100k.tsv'
        }

    def load_imdb_dataset(self, language: str = "english", n_samples: int = 2000) -> pd.DataFrame:
        """Load IMDB movie reviews dataset from local files."""
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
        """Split data into train, dev, and test sets."""
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        n = len(df)
        train_end = int(n * train_ratio)
        dev_end = int(n * (train_ratio + dev_ratio))

        train = df[:train_end]
        dev = df[train_end:dev_end]
        test = df[dev_end:]

        logger.info(f"Split sizes - Train: {len(train)}, Dev: {len(dev)}, Test: {len(test)}")
        return train, dev, test


# ============================================================================
# Preprocessor Class
# ============================================================================

class MultilingualPreprocessor:
    """Preprocessing pipeline for English and Arabic text."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = {
            'english': set(stopwords.words('english')),
            'arabic': set(stopwords.words('arabic'))
        }

        try:
            logger.info("Loading Arabic POS tagger")
            self.arabic_pos_tagger = MLEDisambiguator.pretrained('calima-msa-r13')
            logger.info("Arabic POS tagger loaded")
        except Exception as e:
            logger.error(f"Failed to load Arabic POS tagger: {e}")
            self.arabic_pos_tagger = None

    def clean_text(self, text: str, language: str = 'english') -> str:
        """Clean and normalize text."""
        if language == 'english':
            return self._clean_english(text)
        else:
            return self._clean_arabic(text)

    def _clean_english(self, text: str) -> str:
        """Clean English text."""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\S+@S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _clean_arabic(self, text: str) -> str:
        """Clean Arabic text."""
        text = araby.strip_tashkeel(text)
        text = araby.normalize_hamza(text)
        text = araby.normalize_ligature(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str, language: str = 'english') -> List[str]:
        """Tokenize text into words."""
        if language == 'english':
            return word_tokenize(text)
        else:
            return araby.tokenize(text)

    def sentence_tokenize(self, text: str, language: str = 'english') -> List[str]:
        """Tokenize text into sentences."""
        return sent_tokenize(text, language=language if language == 'english' else 'arabic')

    def remove_stopwords(self, tokens: List[str], language: str = 'english') -> List[str]:
        """Remove stopwords from tokens."""
        stop_words = self.stop_words.get(language, set())
        return [token for token in tokens if token not in stop_words]

    def lemmatize(self, tokens: List[str], language: str = 'english') -> List[str]:
        """Lemmatize tokens."""
        if language == 'english':
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        else:
            return [araby.strip_tashkeel(token) for token in tokens]

    def pos_tag(self, tokens: List[str], language: str = 'english') -> List[Tuple[str, str]]:
        """Perform Part-of-Speech tagging."""
        if language == 'english':
            return nltk_pos_tag(tokens)
        elif language == 'arabic' and self.arabic_pos_tagger:
            try:
                disambiguated_words = self.arabic_pos_tagger.disambiguate(tokens)
                tags = []
                for word in disambiguated_words:
                    if word.analyses:
                        top_analysis = word.analyses[0].analysis
                        pos_tag = top_analysis.get('pos', 'UNK')
                        tags.append((word.word, pos_tag))
                    else:
                        tags.append((word.word, 'UNK'))
                return tags
            except Exception as e:
                logger.error(f"Error during Arabic POS tagging: {e}")
                return [(token, 'UNK') for token in tokens]
        else:
            if language == 'arabic':
                logger.warning("Arabic POS tagger not loaded")
            return [(token, 'UNK') for token in tokens]

    def preprocess_pipeline(self, text: str, language: str = 'english',
                            remove_stops: bool = True,
                            lemmatize: bool = True,
                            do_pos_tag: bool = False) -> Dict:
        """Full preprocessing pipeline."""
        cleaned = self.clean_text(text, language)
        sentences = self.sentence_tokenize(cleaned, language)
        tokens = self.tokenize(cleaned, language)
        full_tokens = tokens

        if remove_stops:
            tokens = self.remove_stopwords(tokens, language)

        if lemmatize:
            tokens = self.lemmatize(tokens, language)

        pos_tags = []
        if do_pos_tag:
            pos_tags = self.pos_tag(full_tokens, language)

        return {
            'original': text,
            'cleaned': cleaned,
            'sentences': sentences,
            'tokens': tokens,
            'pos_tags': pos_tags,
            'n_tokens': len(tokens),
            'n_sentences': len(sentences)
        }


# ============================================================================
# N-gram Language Model Class
# ============================================================================

class NGramLanguageModel:
    """N-gram language model with improved stability and token handling."""

    def __init__(self, n: int = 3, smoothing: str = 'kneser_ney'):
        self.n = n
        self.smoothing = smoothing
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = defaultdict(int)
        self.vocab = set()
        self.vocab_size = 0

    def train(self, corpus: List[List[str]]):
        """Train the language model on a corpus."""
        logger.info(f"Training {self.n}-gram model with {self.smoothing} smoothing")

        corpus = [['<s>'] * (self.n - 1) + sent + ['</s>'] for sent in corpus]

        for sentence in corpus:
            self.vocab.update(sentence)
            for i in range(len(sentence) - self.n + 1):
                ngram = tuple(sentence[i:i + self.n])
                context = ngram[:-1]
                word = ngram[-1]
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1

        self.vocab_size = len(self.vocab)
        logger.info(f"Vocabulary size: {self.vocab_size}")

    def _mle_probability(self, context: Tuple[str], word: str) -> float:
        context_count = self.context_counts.get(context, 0)
        if context_count == 0:
            return 1.0 / self.vocab_size
        ngram_count = self.ngram_counts[context].get(word, 0)
        return ngram_count / context_count

    def _laplace_probability(self, context: Tuple[str], word: str) -> float:
        context_count = self.context_counts.get(context, 0)
        ngram_count = self.ngram_counts[context].get(word, 0)
        return (ngram_count + 1) / (context_count + self.vocab_size)

    def _continuation_probability(self, word: str) -> float:
        contexts_with_word = sum(1 for c in self.ngram_counts if word in self.ngram_counts[c])
        total_contexts = len(self.ngram_counts)
        return contexts_with_word / total_contexts if total_contexts > 0 else 1e-8

    def _kneser_ney_probability(self, context: Tuple[str], word: str, d: float = 0.75) -> float:
        """Kneser-Ney smoothing with floor to prevent zeros."""
        context_count = self.context_counts.get(context, 0)
        if context_count == 0:
            return 1.0 / self.vocab_size
        ngram_count = self.ngram_counts[context].get(word, 0)
        discounted = max(ngram_count - d, 0)
        lambda_weight = d * len(self.ngram_counts[context]) / context_count
        continuation = self._continuation_probability(word)
        return max(discounted / context_count + lambda_weight * continuation, 1e-8)

    def get_probability(self, context: Tuple[str], word: str) -> float:
        if self.smoothing == 'laplace':
            return self._laplace_probability(context, word)
        elif self.smoothing == 'kneser_ney':
            return self._kneser_ney_probability(context, word)
        return self._mle_probability(context, word)

    def calculate_perplexity(self, test_corpus: List[List[str]]) -> float:
        """Stable perplexity computation with padding consistency."""
        log_prob_sum = 0
        word_count = 0
        padded_corpus = []
        for sent in test_corpus:
            if not sent:
                continue
            if sent[0] != '<s>':
                sent = ['<s>'] * (self.n - 1) + sent + ['</s>']
            padded_corpus.append(sent)

        for sentence in padded_corpus:
            for i in range(self.n - 1, len(sentence)):
                context = tuple(sentence[i - self.n + 1:i])
                word = sentence[i]
                prob = self.get_probability(context, word)
                log_prob_sum += np.log2(prob)
                word_count += 1

        avg_log_prob = log_prob_sum / max(word_count, 1)
        perplexity = 2 ** (-avg_log_prob)
        return perplexity


# ============================================================================
# Named Entity Recognition Class
# ============================================================================

class NamedEntityRecognizer:
    """Named Entity Recognition for movie domain."""

    def __init__(self, language: str = 'english'):
        self.language = language
        self.gazetteers = self._load_gazetteers()

    def _load_gazetteers(self) -> Dict[str, set]:
        """Load domain-specific gazetteers."""
        gazetteers = {
            'ACTOR': {
                'Leonardo DiCaprio', 'Meryl Streep', 'Tom Hanks',
                'Scarlett Johansson', 'Denzel Washington', 'Brad Pitt',
                'ليوناردو دي كابريو', 'ميريل ستريب', 'توم هانكس'
            },
            'DIRECTOR': {
                'Christopher Nolan', 'Steven Spielberg', 'Martin Scorsese',
                'Quentin Tarantino', 'James Cameron',
                'كريستوفر نولان', 'ستيفن سبيلبرغ', 'مارتن سكورسيزي'
            },
            'MOVIE': {
                'Inception', 'The Departed', 'Interstellar', 'Titanic',
                'Pulp Fiction', 'Schindler\'s List',
                'إنسبشن', 'تايتانك', 'قائمة شندلر'
            },
            'STUDIO': {
                'Warner Bros', 'Universal Pictures', 'Paramount',
                'Disney', 'Sony Pictures', '20th Century Fox'
            }
        }
        return gazetteers

    def extract_entities(self, text: str, tokens: List[str]) -> List[Tuple[str, str, int, int]]:
        """Extract named entities from text."""
        entities = []
        for entity_type, names in self.gazetteers.items():
            for name in names:
                pattern = re.escape(name)
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append((
                        match.group(),
                        entity_type,
                        match.start(),
                        match.end()
                    ))
        entities = sorted(set(entities), key=lambda x: x[2])
        return entities


# ============================================================================
# Baseline Parser (Chunker)
# ============================================================================

class BaselineParser:
    """Baseline parser using NLTK's RegexpParser for chunking."""

    def __init__(self, language: str = 'english'):
        self.language = language
        self.grammar = self._define_grammar()

    def _define_grammar(self) -> str:
        """Define chunking grammar."""
        if self.language == 'english':
            return r"NP: {<DT>?<JJ.*>*<NN.*>+}"
        else:
            logger.warning(f"No chunking grammar defined for {self.language}")
            return ""

    def chunk(self, pos_tagged_tokens: List[Tuple[str, str]]) -> Tree:
        """Apply chunking to POS-tagged tokens."""
        if not self.grammar:
            return Tree('S', pos_tagged_tokens)
        try:
            cp = RegexpParser(self.grammar)
            tree = cp.parse(pos_tagged_tokens)
            return tree
        except Exception as e:
            logger.error(f"Error during parsing: {e}")
            return Tree('S', pos_tagged_tokens)


# ============================================================================
# Sentiment Classifier Class
# ============================================================================

from sklearn.preprocessing import LabelEncoder


class SentimentClassifier:
    """Ensemble sentiment classifier for movie reviews."""

    def __init__(self, language: str = 'english'):
        self.language = language
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8
        )
        self.label_encoder = LabelEncoder()
        self.nb_classifier = MultinomialNB(alpha=0.1)
        self.svm_classifier = LinearSVC(C=1.0, max_iter=1000, dual='auto')
        self.xgb_classifier = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    def train(self, X_train: List[str], y_train: List[str]):
        """Train the sentiment classifier."""
        logger.info(f"Training sentiment classifier for {self.language}")
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        X_train_vec = self.vectorizer.fit_transform(X_train)

        self.nb_classifier.fit(X_train_vec, y_train)
        self.svm_classifier.fit(X_train_vec, y_train)
        X_train_dense = X_train_vec.toarray()
        self.xgb_classifier.fit(X_train_dense, y_train_encoded)

        logger.info("Training completed")

    def predict(self, X_test: List[str], use_ensemble: bool = True) -> np.ndarray:
        """Predict sentiment for test data."""
        X_test_vec = self.vectorizer.transform(X_test)

        if use_ensemble:
            nb_pred = self.nb_classifier.predict(X_test_vec)
            svm_pred = self.svm_classifier.predict(X_test_vec)
            xgb_pred_numeric = self.xgb_classifier.predict(X_test_vec.toarray())
            xgb_pred = self.label_encoder.inverse_transform(xgb_pred_numeric)

            predictions = []
            for nb, svm, xgb in zip(nb_pred, svm_pred, xgb_pred):
                votes = [nb, svm, xgb]
                predictions.append(max(set(votes), key=votes.count))
            return np.array(predictions)
        else:
            return self.svm_classifier.predict(X_test_vec)

    def evaluate(self, X_test: List[str], y_test: List[str]) -> Dict:
        """Evaluate the classifier."""
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average='weighted', zero_division=0
        )
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(y_test, predictions, zero_division=0)
        }


# ============================================================================
# Main Pipeline Execution
# ============================================================================

def run_complete_pipeline():
    """Main pipeline execution."""
    logger.info("MULTILINGUAL MOVIE REVIEWS NLP PIPELINE")

    data_loader = MultilingualDataLoader()
    preprocessor = MultilingualPreprocessor()
    results = {}

    for language in ['english', 'arabic']:
        logger.info(f"\nProcessing {language.upper()} reviews")
        results[language] = {}

        df = data_loader.load_imdb_dataset(language=language, n_samples=1200)
        if df.empty:
            logger.error(f"Failed to load data for {language}. Skipping.")
            continue
        train_df, dev_df, test_df = data_loader.create_train_test_split(df)

        logger.info("Preprocessing data")
        train_texts = train_df['text'].tolist()
        train_labels = train_df['sentiment'].tolist()

        test_texts = test_df['text'].tolist()
        test_labels = test_df['sentiment'].tolist()

        logger.info("Performing POS Tagging and Parsing")
        parser = BaselineParser(language=language)
        sample_text = train_texts[0]
        sample_tokens_full = preprocessor.tokenize(
            preprocessor.clean_text(sample_text, language), language
        )
        sample_pos_tags = preprocessor.pos_tag(sample_tokens_full, language)
        chunk_tree = parser.chunk(sample_pos_tags)

        logger.info(f"Sample POS tags (first 10): {sample_pos_tags[:10]}")
        logger.info(f"Sample Parse Tree:\n{chunk_tree}")

        logger.info(f"Calculating POS tag frequencies for {language} test set")
        all_tags = []
        test_tokens_list = [
            preprocessor.tokenize(preprocessor.clean_text(text, language), language)
            for text in test_texts
        ]

        for tokens in test_tokens_list:
            if tokens:
                tags = preprocessor.pos_tag(tokens, language)
                all_tags.extend([tag for word, tag in tags])

        tag_counts = Counter(all_tags)
        top_10_tags = tag_counts.most_common(10)
        logger.info(f"Top 5 {language} POS tags: {top_10_tags[:5]}")

        results[language]['pos_tagging'] = {
            'sample_tags': sample_pos_tags[:10],
            'top_10_tags': top_10_tags
        }
        results[language]['parsing'] = {'sample_tree_str': str(chunk_tree)}

        logger.info("Training language models")
        lm_results = {}

        def normalize_and_tokenize(text, language):
            cleaned = preprocessor.clean_text(text, language)
            tokens = preprocessor.tokenize(cleaned, language)
            if language == 'arabic':
                tokens = [t for t in tokens if re.match(r'^[\u0621-\u064A]+$', t)]
            return tokens

        for n in [2, 3, 4]:
            lm = NGramLanguageModel(n=n, smoothing='kneser_ney')
            train_corpus = [normalize_and_tokenize(text, language) for text in train_texts[:200]]
            test_corpus = [normalize_and_tokenize(text, language) for text in test_df['text'].tolist()[:50]]

            lm.train(train_corpus)
            perplexity = lm.calculate_perplexity(test_corpus)

            lm_results[f'{n}-gram'] = {
                'perplexity': perplexity,
                'vocab_size': lm.vocab_size
            }
            logger.info(f"{n}-gram perplexity: {perplexity:.2f}")

        logger.info("Training sentiment classifier")
        sentiment_clf = SentimentClassifier(language=language)
        sentiment_clf.train(train_texts, train_labels)

        sentiment_metrics = sentiment_clf.evaluate(test_texts, test_labels)

        logger.info(f"Accuracy: {sentiment_metrics['accuracy']:.4f}")
        logger.info(f"F1-Score: {sentiment_metrics['f1_score']:.4f}")

        logger.info("Performing Named Entity Recognition")
        ner = NamedEntityRecognizer(language=language)
        sample_text_ner = test_texts[0]
        sample_tokens_ner = preprocessor.tokenize(sample_text_ner, language)
        entities = ner.extract_entities(sample_text_ner, sample_tokens_ner)
        logger.info(f"Sample NER Results: {len(entities)} entities found")

        results[language]['language_model'] = lm_results
        results[language]['sentiment_analysis'] = {
            'accuracy': sentiment_metrics['accuracy'],
            'precision': sentiment_metrics['precision'],
            'recall': sentiment_metrics['recall'],
            'f1_score': sentiment_metrics['f1_score']
        }
        results[language]['ner'] = {'sample_entities': len(entities)}

    logger.info("\nFINAL RESULTS SUMMARY")
    for lang, metrics in results.items():
        logger.info(f"\n{lang.upper()}:")
        logger.info(f"  Best Perplexity: {min(m['perplexity'] for m in metrics['language_model'].values()):.2f}")
        logger.info(f"  Sentiment Accuracy: {metrics['sentiment_analysis']['accuracy']:.4f}")
        logger.info(f"  Sentiment F1: {metrics['sentiment_analysis']['f1_score']:.4f}")

    return results


results = run_complete_pipeline()

# ============================================================================
# Visualize Results
# ============================================================================

languages = list(results.keys())
accuracies = [results[lang]['sentiment_analysis']['accuracy'] for lang in languages]
f1_scores = [results[lang]['sentiment_analysis']['f1_score'] for lang in languages]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(languages))
width = 0.35
axes[0].bar(x - width / 2, accuracies, width, label='Accuracy', color='#3b82f6')
axes[0].bar(x + width / 2, f1_scores, width, label='F1-Score', color='#10b981')
axes[0].set_xlabel('Language')
axes[0].set_ylabel('Score')
axes[0].set_title('Sentiment Analysis Performance by Language')
axes[0].set_xticks(x)
axes[0].set_xticklabels([lang.title() for lang in languages])
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim([0, 1])

english_lm = results['english']['language_model']
arabic_lm = results['arabic']['language_model']
ngrams = list(english_lm.keys())
en_perplexities = [english_lm[ng]['perplexity'] for ng in ngrams]
ar_perplexities = [arabic_lm[ng]['perplexity'] for ng in ngrams]

axes[1].plot(ngrams, en_perplexities, marker='o', linewidth=2, label='English', color='#3b82f6')
axes[1].plot(ngrams, ar_perplexities, marker='s', linewidth=2, label='Arabic', color='#10b981')
axes[1].set_xlabel('N-gram Model')
axes[1].set_ylabel('Perplexity')
axes[1].set_title('Language Model Perplexity Comparison')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')

plt.tight_layout()
plt.show()

# ============================================================================
# MODIFIED VISUALIZATION SECTION
# ============================================================================

fig_pos, axes_pos = plt.subplots(1, 2, figsize=(14, 6))
fig_pos.suptitle('Top 10 Most Common POS Tags by Language (from Test Set)', fontsize=16)

try:
    en_tags = results['english']['pos_tagging']['top_10_tags']
    if en_tags:
        labels, counts = zip(*en_tags)
        labels = list(labels)
        counts = list(counts)
        ticks = np.arange(len(labels))

        # Changed from barplot to scatterplot
        sns.scatterplot(
            x=counts,
            y=ticks,
            ax=axes_pos[0],
            color='#3b82f6',
            size=counts,  # Use count for bubble size
            sizes=(100, 1000),  # Define min/max bubble size
            legend=False  # Hide the size legend
        )

        axes_pos[0].set_yticks(ticks)
        axes_pos[0].set_yticklabels(labels)
        axes_pos[0].set_title('English POS Tag Frequencies (Scatter Plot)')
        axes_pos[0].set_xlabel('Total Count in Test Set')
        axes_pos[0].grid(True, alpha=0.3)
    else:
        axes_pos[0].text(0.5, 0.5, 'No English POS tags found.', horizontalalignment='center',
                         verticalalignment='center', transform=axes_pos[0].transAxes)
except Exception as e:
    logger.error(f"Could not plot English POS tags: {e}")
    axes_pos[0].text(0.5, 0.5, 'Error plotting English tags.', horizontalalignment='center', verticalalignment='center',
                     transform=axes_pos[0].transAxes)

try:
    ar_tags = results['arabic']['pos_tagging']['top_10_tags']
    if ar_tags and ar_tags[0][0] != 'UNK':
        labels, counts = zip(*ar_tags)
        labels = list(labels)
        counts = list(counts)

        # Set font that supports Arabic glyphs
        plt.rcParams['font.family'] = ['Arial', 'sans-serif']
        ticks = np.arange(len(labels))

        # Changed from barplot to scatterplot
        sns.scatterplot(
            x=counts,
            y=ticks,
            ax=axes_pos[1],
            color='#10b981',
            size=counts,  # Use count for bubble size
            sizes=(100, 1000),  # Define min/max bubble size
            legend=False
        )

        axes_pos[1].set_yticks(ticks)
        axes_pos[1].set_yticklabels(labels)
        axes_pos[1].set_title('Arabic POS Tag Frequencies (Scatter Plot)')
        axes_pos[1].set_xlabel('Total Count in Test Set')
        axes_pos[1].grid(True, alpha=0.3)
    else:
        axes_pos[1].text(0.5, 0.5, 'Arabic POS Tagger not run or only found "UNK" tags.', horizontalalignment='center',
                         verticalalignment='center', transform=axes_pos[1].transAxes)
except Exception as e:
    logger.error(f"Could not plot Arabic POS tags: {e}")
    axes_pos[1].text(0.5, 0.5, 'Error plotting Arabic tags.', horizontalalignment='center', verticalalignment='center',
                     transform=axes_pos[1].transAxes)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ============================================================================
# END OF MODIFIED SECTION
# ============================================================================


# ============================================================================
# Test Individual Components
# ============================================================================

logger.info("TESTING INDIVIDUAL COMPONENTS")

preprocessor = MultilingualPreprocessor()
ner = NamedEntityRecognizer('english')
parser = BaselineParser('english')

logger.info("1. PREPROCESSING TEST")
test_text = "This movie was absolutely fantastic! Christopher Nolan's direction was brilliant."
result = preprocessor.preprocess_pipeline(test_text, 'english', do_pos_tag=True)

logger.info(f"Original: {result['original']}")
logger.info(f"Cleaned: {result['cleaned']}")
logger.info(f"Tokens: {result['tokens'][:10]}")
logger.info(f"POS Tags: {result['pos_tags'][:10]}")
logger.info(f"Number of tokens: {result['n_tokens']}")
logger.info(f"Number of sentences: {result['n_sentences']}")

logger.info("2. NAMED ENTITY RECOGNITION TEST")
test_text2 = "Leonardo DiCaprio starred in Inception directed by Christopher Nolan."
entities = ner.extract_entities(test_text2, test_text2.split())

logger.info(f"Text: {test_text2}")
logger.info(f"Entities found: {len(entities)}")
for entity, entity_type, start, end in entities:
    logger.info(f"  - {entity} [{entity_type}]")

logger.info("3. PARSING TEST")
test_text3 = "The talented actor gave a great performance in the new movie."
tokens3 = preprocessor.tokenize(preprocessor.clean_text(test_text3, 'english'), 'english')
tags3 = preprocessor.pos_tag(tokens3, 'english')
tree3 = parser.chunk(tags3)

logger.info(f"Text: {test_text3}")
logger.info(f"Parse Tree:\n{tree3}")

# ============================================================================
# Display Summary Statistics
# ============================================================================

logger.info("DETAILED STATISTICS")

logger.info("DATASET STATISTICS:")
for language in ['english', 'arabic']:
    df = MultilingualDataLoader().load_imdb_dataset(language, 1200)
    logger.info(f"\n{language.upper()}:")
    logger.info(f"  Total reviews: {len(df)}")
    logger.info(f"  Positive: {(df['sentiment'] == 'positive').sum()}")
    logger.info(f"  Negative: {(df['sentiment'] == 'negative').sum()}")
    logger.info(f"  Avg review length: {df['text'].str.split().str.len().mean():.1f} words")

logger.info("\nMODEL PERFORMANCE:")
for lang in ['english', 'arabic']:
    metrics = results[lang]['sentiment_analysis']
    logger.info(f"\n{lang.upper()}:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")

logger.info("\nLANGUAGE MODEL PERPLEXITY:")
for lang in ['english', 'arabic']:
    lm_metrics = results[lang]['language_model']
    logger.info(f"\n{lang.upper()}:")
    for ngram, metrics in lm_metrics.items():
        logger.info(f"  {ngram}: {metrics['perplexity']:.2f} (vocab: {metrics['vocab_size']})")

logger.info("\nBASELINE COMPONENTS:")
for lang in ['english', 'arabic']:
    logger.info(f"\n{lang.upper()}:")
    logger.info(f"  POS Tagging: {results[lang]['pos_tagging']['sample_tags'][0]}")
    parsing_result = 'Found Noun Phrases' if '(NP' in results[lang]['parsing']['sample_tree_str'] else 'No Chunks'
    logger.info(f"  Parsing: {parsing_result}")

logger.info("\nCROSS-LANGUAGE COMPARISON:")
en_acc = results['english']['sentiment_analysis']['accuracy']
ar_acc = results['arabic']['sentiment_analysis']['accuracy']
diff = en_acc - ar_acc

logger.info(f"English Accuracy:  {en_acc:.4f}")
logger.info(f"Arabic Accuracy:   {ar_acc:.4f}")
logger.info(f"Difference:        {diff:.4f}")
logger.info(f"Average:           {(en_acc + ar_acc) / 2:.4f}")

if diff > 0:
    logger.info(f"English performs {abs(diff) * 100:.2f}% better")
else:
    logger.info(f"Arabic performs {abs(diff) * 100:.2f}% better")

# ============================================================================
# Error Analysis
# ============================================================================

logger.info("ERROR ANALYSIS")

data_loader = MultilingualDataLoader()
preprocessor = MultilingualPreprocessor()

df_en = data_loader.load_imdb_dataset('english', 200)
train_df, dev_df, test_df = data_loader.create_train_test_split(df_en)

sentiment_clf = SentimentClassifier('english')
sentiment_clf.train(train_df['text'].tolist(), train_df['sentiment'].tolist())

X_test = test_df['text'].tolist()
y_test = test_df['sentiment'].tolist()
y_pred = sentiment_clf.predict(X_test)

logger.info("MISCLASSIFIED EXAMPLES:")
error_count = 0
for i, (text, true, pred) in enumerate(zip(X_test, y_test, y_pred)):
    if true != pred and error_count < 5:
        error_count += 1
        logger.info(f"\nError #{error_count}:")
        logger.info(f"  Review: {text[:100]}...")
        logger.info(f"  True Sentiment: {true}")
        logger.info(f"  Predicted: {pred}")

total_errors = sum(1 for true, pred in zip(y_test, y_pred) if true != pred)
error_rate = total_errors / len(y_test)

logger.info(f"\nERROR STATISTICS:")
logger.info(f"  Total test samples: {len(y_test)}")
logger.info(f"  Correct predictions: {len(y_test) - total_errors}")
logger.info(f"  Errors: {total_errors}")
logger.info(f"  Error rate: {error_rate:.2%}")
logger.info(f"  Accuracy: {1 - error_rate:.2%}")

# ============================================================================
# Save Results
# ============================================================================

import json
from datetime import datetime

logger.info("SAVING RESULTS")

save_data = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'project': 'Multilingual Movie Reviews NLP Pipeline',
    'course': 'CSAI411 - Introduction to Natural Language Processing',
    'results': results,
    'summary': {
        'languages': ['english', 'arabic'],
        'total_reviews': 2400,
        'average_sentiment_accuracy': np.mean([
            results['english']['sentiment_analysis']['accuracy'],
            results['arabic']['sentiment_analysis']['accuracy']
        ]),
        'best_perplexity': min(
            min(m['perplexity'] for m in results['english']['language_model'].values()),
            min(m['perplexity'] for m in results['arabic']['language_model'].values())
        )
    }
}

with open('pipeline_results.json', 'w', encoding='utf-8') as f:
    json.dump(save_data, f, indent=2, ensure_ascii=False)

logger.info("Results saved to 'pipeline_results.json'")

logger.info("FINAL SUMMARY")
logger.info(f"Total Reviews Processed: {save_data['summary']['total_reviews']}")
logger.info(f"Average Sentiment Accuracy: {save_data['summary']['average_sentiment_accuracy']:.2%}")
logger.info(f"Best Language Model Perplexity: {save_data['summary']['best_perplexity']:.2f}")
logger.info(f"Languages Supported: {', '.join(save_data['summary']['languages']).title()}")


# ============================================================================
# Quick Test Function
# ============================================================================

def analyze_review(text, language='english'):
    """Quick function to analyze a single review."""
    logger.info("REVIEW ANALYSIS")

    preprocessor = MultilingualPreprocessor()
    processed = preprocessor.preprocess_pipeline(text, language, do_pos_tag=True)

    logger.info(f"Original Review: {text}")
    logger.info(f"\nTokens ({processed['n_tokens']} total): {', '.join(processed['tokens'][:15])}...")
    logger.info(f"Sentences: {processed['n_sentences']}")
    logger.info(f"\nPOS Tags (first 10): {processed['pos_tags'][:10]}")

    parser = BaselineParser(language)
    chunk_tree = parser.chunk(processed['pos_tags'])
    logger.info(f"\nNoun Phrases:")

    np_chunks = []
    for subtree in chunk_tree.subtrees():
        if subtree.label() == 'NP':
            np_chunks.append(" ".join(word for word, tag in subtree.leaves()))

    if np_chunks:
        for chunk in np_chunks:
            logger.info(f"  - {chunk}")
    else:
        logger.info("  None detected")

    ner = NamedEntityRecognizer(language)
    entities = ner.extract_entities(text, processed['tokens'])

    logger.info(f"\nNamed Entities ({len(entities)} found):")
    if entities:
        for entity, entity_type, _, _ in entities:
            logger.info(f"  - {entity} [{entity_type}]")
    else:
        logger.info("  None detected")

    positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'brilliant', 'superb'}
    negative_words = {'bad', 'terrible', 'awful', 'poor', 'horrible', 'disappointing', 'boring', 'confusing'}

    tokens_lower = [t.lower() for t in processed['tokens']]
    pos_count = sum(1 for t in tokens_lower if t in positive_words)
    neg_count = sum(1 for t in tokens_lower if t in negative_words)

    if pos_count > neg_count:
        sentiment = 'POSITIVE'
        confidence = pos_count / (pos_count + neg_count) if (pos_count + neg_count) > 0 else 0.5
    elif neg_count > pos_count:
        sentiment = 'NEGATIVE'
        confidence = neg_count / (pos_count + neg_count) if (pos_count + neg_count) > 0 else 0.5
    else:
        sentiment = 'NEUTRAL'
        confidence = 0.5

    logger.info(f"\nSentiment: {sentiment}")
    logger.info(f"Confidence: {confidence:.2%}")
    logger.info(f"Positive indicators: {pos_count}")
    logger.info(f"Negative indicators: {neg_count}")

    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'tokens': processed['tokens'],
        'entities': entities
    }


logger.info("TEST EXAMPLES:")

logger.info("Test 1:")
analyze_review(
    "Christopher Nolan's Inception was absolutely brilliant! Leonardo DiCaprio delivered an outstanding performance.",
    "english"
)

logger.info("\nTest 2:")
analyze_review(
    "The movie was terrible and boring. Even Tom Hanks couldn't save this disaster.",
    "english"
)

logger.info("\nPipeline execution complete.")