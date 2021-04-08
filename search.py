import re

import string

import joblib

import numpy as np

from scipy.sparse import save_npz, load_npz

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import linear_kernel

from botok import Text


def tokenize(text):
    # semantic

    try:
        # https://github.com/Esukhia/botok/issues/18

        text = Text(text).tokenize_words_raw_text
    except IndexError:
        text = ''.join(text.split())

        text = Text(text).tokenize_words_raw_text

    words = []

    # split by punctuation (also will be removed) and empty spaces

    for word in re.split('[\s' + string.punctuation + ']+', text):
        if not word:
            continue

        # remove numeric characters

        if word.isnumeric():
            continue

        words.append(word)

    return words


class SearchEngine:
    def __init__(self, corpus=None, stop_words_txt='data/stop_words.txt', model_path='data/tfidf.joblib', feats_path='data/tfidf.npz'):
        if corpus:
            # stop words will be removed from the resulting tokens

            with open(stop_words_txt, encoding='utf-8') as f:
                stop_words = f.read().split('\n')

            self.vectorizer = TfidfVectorizer(
                tokenizer=tokenize,
                stop_words=stop_words,
                max_df=0.7,
                # max_features=1000,
            )

            self.features = self.vectorizer.fit_transform(corpus)

            # save model & feats

            joblib.dump(self.vectorizer, model_path)

            save_npz(feats_path, self.features)
        else:
            # load model & feats

            self.vectorizer = joblib.load(model_path)

            self.features = load_npz(feats_path)

    def __call__(self, query_text, min_score=0.05):
        query_feats = self.vectorizer.transform([query_text])

        # equivalent to cosine_similarity

        similarities = linear_kernel(query_feats, self.features)[0]

        # sorted in descending order

        sorted_index = similarities.argsort()[::-1]

        sorted_similarities = similarities[sorted_index]

        # ignore completely irrelevant ones

        sorted_index = sorted_index[sorted_similarities > min_score, np.newaxis]

        sorted_similarities = sorted_similarities[sorted_similarities > min_score, np.newaxis]

        return np.hstack([
            sorted_index,
            sorted_similarities,
        ])
