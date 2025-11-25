import nltk
from typing import List, Tuple
from nltk import RegexpParser, Tree, pos_tag  
from nltk.tokenize import word_tokenize
from config import logger


class NamedEntityRecognizer:
    def __init__(self, language: str = 'english'):
        self.language = language
        try:
            nltk.data.find('chunkers/maxent_ne_chunker_tab')
            nltk.data.find('words')
        except LookupError:
            nltk.download('maxent_ne_chunker_tab', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)  
            nltk.download('words', quiet=True)

    def extract_entities(self, text: str, tokens: List[str] = None) -> List[Tuple[str, str, int, int]]:
        if self.language != 'english':
            return []

        if not tokens:
            tokens = word_tokenize(text)

        tags = pos_tag(tokens)

        try:
            chunks = nltk.ne_chunk(tags)
        except Exception as e:
            logger.warning(f"NER failed: {e}")
            return []

        entities = []
        current_pos = 0

        for chunk in chunks:
            if hasattr(chunk, 'label'):
                entity_name = " ".join(c[0] for c in chunk)
                entity_type = chunk.label()

                start = text.find(entity_name, current_pos)
                if start != -1:
                    end = start + len(entity_name)
                    entities.append((entity_name, entity_type, start, end))
                    current_pos = end
            else:
                word = chunk[0]
                next_pos = text.find(word, current_pos)
                if next_pos != -1:
                    current_pos = next_pos + len(word)

        return entities

class BaselineParser:

    def __init__(self, language: str = 'english'):
        self.language = language
        self.grammar = self._define_grammar()

    def _define_grammar(self) -> str:
        if self.language == 'english':
            return r"NP: {<DT>?<JJ.*>*<NN.*>+}"
        elif self.language == 'arabic':
            return r"NP: {<noun.*|noun_prop>+<adj.*>*}"
        else:
            logger.warning(f"No chunking grammar defined for {self.language}")
            return ""

    def chunk(self, pos_tagged_tokens: List[Tuple[str, str]]) -> Tree:
        if not self.grammar:
            return Tree('S', pos_tagged_tokens)
        try:
            cp = RegexpParser(self.grammar)
            tree = cp.parse(pos_tagged_tokens)
            return tree
        except Exception as e:
            logger.error(f"Error during parsing: {e}")
            return Tree('S', pos_tagged_tokens)