import pickle

import pandas as pd
import pytest

from src.language_model_usage.winograd_schema_challenge import find_missing_wsc_words_in_corpus_vocab


@pytest.fixture()
def corpus():
    corpus_file_name = 'tests/test_language_model_usage/mock_data/corpus.pkl'
    corpus = pickle.load(open(corpus_file_name, "rb"))

    return corpus


@pytest.fixture()
def df():
    df = pd.DataFrame({
        'text_col_1': ['text11 text22', 'text11 text21'],
        'text_col_2': ['missing text21', 'words text41'],
        'non_text_col': [True, False],
    })

    return df


class TestFindMissingWscWordsInCorpusVocab:
    def test_find_missing_wsc_words_in_corpus_vocab(self, corpus, df):
        text_columns = df.loc[:, (df.applymap(type) == str).all(0)].columns

        missing_words = find_missing_wsc_words_in_corpus_vocab(df, text_columns, corpus, english=False)

        assert set(missing_words) == set(['missing', 'words'])

    def test_find_missing_wsc_words_in_corpus_vocab_english(self, corpus, df):
        text_columns = df.loc[:, (df.applymap(type) == str).all(0)].columns

        missing_words = find_missing_wsc_words_in_corpus_vocab(df, text_columns, corpus, english=True)

        assert set(missing_words) == set(['missing', 'words'])
