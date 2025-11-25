import re
import nltk
from typing import List, Dict, Tuple  
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag as nltk_pos_tag
import pyarabic.araby as araby
from camel_tools.disambig.mle import MLEDisambiguator
from config import logger

# NLTK Downloads
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

class MultilingualPreprocessor:
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
        if language == 'english':
            return self._clean_english(text)
        else:
            return self._clean_arabic(text)

    def _clean_english(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\S+@S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _clean_arabic(self, text: str) -> str:
        text = araby.strip_tashkeel(text)
        text = araby.normalize_hamza(text)
        text = araby.normalize_ligature(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str, language: str = 'english') -> List[str]:
        if language == 'english':
            return word_tokenize(text)
        else:
            return araby.tokenize(text)

    def sentence_tokenize(self, text: str, language: str = 'english') -> List[str]:
        return sent_tokenize(text, language=language if language == 'english' else 'arabic')

    def remove_stopwords(self, tokens: List[str], language: str = 'english') -> List[str]:
        stop_words = self.stop_words.get(language, set())
        return [token for token in tokens if token not in stop_words]

    def lemmatize(self, tokens: List[str], language: str = 'english') -> List[str]:
        if language == 'english':
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        else:
            return [araby.strip_tashkeel(token) for token in tokens]

    def pos_tag(self, tokens: List[str], language: str = 'english') -> List[Tuple[str, str]]:
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