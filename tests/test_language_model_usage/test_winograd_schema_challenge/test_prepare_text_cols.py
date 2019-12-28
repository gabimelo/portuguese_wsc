import pickle
from unittest import mock

import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from src.language_model_usage.winograd_schema_challenge import prepare_text_cols


@pytest.fixture()
def corpus():
    corpus_file_name = 'tests/test_language_model_usage/mock_data/corpus.pkl'
    corpus = pickle.load(open(corpus_file_name, "rb"))

    return corpus


@pytest.fixture()
def df_pt():
    df = pd.DataFrame({
        'text_col_1': ['``text11`` text22', 'text11 text21'],
        'text_col_2': ['missing text21', 'words d\'agua'],
        'non_text_col': [True, False],
    })

    return df


@pytest.fixture()
def df_en():
    df = pd.DataFrame({
        'text_col_1': ['``text1.1`` text2.2', 'text1.1 text2.1'],
        'text_col_2': ['missing text2.1', 'words don\'t'],
        'non_text_col': [True, False],
    })

    return df


@mock.patch(
    'src.language_model_usage.winograd_schema_challenge.find_missing_wsc_words_in_corpus_vocab',
    return_value=['faltante', 'missing']
)
class TestPrepareTextCols:
    def test_prepare_text_cols(self, corpus, df_pt):
        df_result = prepare_text_cols(df_pt, corpus, False)

        assert_array_equal(df_result.text_col_1, ['`` text11 `` text22', 'text11 text21'])
        assert_array_equal(df_result.text_col_2, ['<UNK> text21', 'words d\'agua'])
        assert_array_equal(df_result.non_text_col, [True, False])

    def test_prepare_text_cols_english(self, corpus, df_en):
        df_result = prepare_text_cols(df_en, corpus, True)

        assert_array_equal(df_result.text_col_1, ['" text1 @.@ 1 " text2 @.@ 2', 'text1 @.@ 1 text2 @.@ 1'])
        assert_array_equal(df_result.text_col_2, ['<UNK> text2 @.@ 1', 'words don \'t'])
        assert_array_equal(df_result.non_text_col, [True, False])
