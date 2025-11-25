import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from config import logger
from typing import List, Dict, Tuple 

class NGramLanguageModel:
    def __init__(self, n: int = 3, smoothing: str = 'kneser_ney'):
        self.n = n
        self.smoothing = smoothing
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = defaultdict(int)
        self.vocab = set()
        self.vocab_size = 0

    def train(self, corpus: List[List[str]]):
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
    
class SentimentClassifier:
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
            eval_metric='logloss'
        )

    def train(self, X_train: List[str], y_train: List[str]):
        logger.info(f"Training sentiment classifier for {self.language}")
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        X_train_vec = self.vectorizer.fit_transform(X_train)

        self.nb_classifier.fit(X_train_vec, y_train)
        self.svm_classifier.fit(X_train_vec, y_train)
        X_train_dense = X_train_vec.toarray()
        self.xgb_classifier.fit(X_train_dense, y_train_encoded)

        logger.info("Training completed")

    def predict(self, X_test: List[str], use_ensemble: bool = True) -> np.ndarray:
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