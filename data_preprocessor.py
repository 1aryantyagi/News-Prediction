import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors


class DataPreprocessor:
    def __init__(self, word2vec_path, max_seq_length=300):
        self.stop_words = set(stopwords.words('english'))
        self.word2vec = KeyedVectors.load_word2vec_format(
            word2vec_path, binary=True)
        self.max_seq_length = max_seq_length
        self.embedding_dim = self.word2vec.vector_size

    def clean_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
        text = text.lower().strip()
        tokens = word_tokenize(text)
        filtered_tokens = [w for w in tokens if w not in self.stop_words]
        return filtered_tokens

    def text_to_vectors(self, text):
        tokens = self.clean_text(text)
        vectors = []
        for token in tokens:
            if token in self.word2vec:
                vectors.append(self.word2vec[token])

        if len(vectors) > self.max_seq_length:
            vectors = vectors[:self.max_seq_length]
        else:
            vectors += [np.zeros(self.embedding_dim)] * \
                (self.max_seq_length - len(vectors))
        return np.array(vectors)
