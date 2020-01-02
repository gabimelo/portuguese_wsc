from unittest import mock

import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from src.language_model_usage.winograd_schema_challenge import run_test_for_col


@pytest.fixture()
def df():
    df = pd.DataFrame({
        'correct_sentence':                  ['text', 'text2'],  # noqa: E241
        'incorrect_sentence':                ['text', 'text2'],  # noqa: E241
        'correct_switched':                  ['text', 'text2'],  # noqa: E241
        'incorrect_switched':                ['text', 'text2'],  # noqa: E241
        'manually_fixed_correct_sentence':   ['text', 'text2'],  # noqa: E241
        'manually_fixed_incorrect_sentence': ['text', 'text2'],  # noqa: E241
        'is_associative':                    [True,  False],  # noqa: E241
        'is_switchable':                     [True,  False],  # noqa: E241
        'original_result_full':              [False, False],  # noqa: E241
        'original_result_partial':           [False, False],  # noqa: E241
        'switched_result_full':              [False, False],  # noqa: E241
        'switched_result_partial':           [False, False],  # noqa: E241
        'manually_fixed_result_full':        [False, False],  # noqa: E241
        'manually_fixed_result_partial':     [False, False],  # noqa: E241
    })

    return df


count = 0


def mock_analyse_single_wsc(model_file_name, corpus, device, correct_sentence, wrong_sentence, model):
    global count
    count += 1

    if count % 2 == 0:
        return False, True
    if count % 2 == 1:
        return True, False


@mock.patch(
    'src.language_model_usage.winograd_schema_challenge.analyse_single_wsc',
    side_effect=mock_analyse_single_wsc
)
class TestRunTestForCol:
    def test_run_test_for_original(self, mocked_analyse_single_wsc, df):
        df_result = run_test_for_col(df, 'model', 'model_file_name', 'corpus', 'device', 'original')

        assert_array_equal(df_result['original_result_full'], [True, False])
        assert_array_equal(df_result['original_result_partial'], [False, True])
        assert_array_equal(df_result['switched_result_full'], [False, False])
        assert_array_equal(df_result['switched_result_partial'], [False, False])
        assert_array_equal(df_result['manually_fixed_result_full'], [False, False])
        assert_array_equal(df_result['manually_fixed_result_partial'], [False, False])

    def test_run_test_for_switched(self, mocked_analyse_single_wsc, df):
        df_result = run_test_for_col(df, 'model', 'model_file_name', 'corpus', 'device', 'switched')

        assert_array_equal(df_result['original_result_full'], [False, False])
        assert_array_equal(df_result['original_result_partial'], [False, False])
        assert_array_equal(df_result['switched_result_full'], [True, False])
        assert_array_equal(df_result['switched_result_partial'], [False, True])
        assert_array_equal(df_result['manually_fixed_result_full'], [False, False])
        assert_array_equal(df_result['manually_fixed_result_partial'], [False, False])

    def test_run_test_for_manually_fixed(self, mocked_analyse_single_wsc, df):
        df_result = run_test_for_col(df, 'model', 'model_file_name', 'corpus', 'device', 'manually_fixed')

        assert_array_equal(df_result['original_result_full'], [False, False])
        assert_array_equal(df_result['original_result_partial'], [False, False])
        assert_array_equal(df_result['switched_result_full'], [False, False])
        assert_array_equal(df_result['switched_result_partial'], [False, False])
        assert_array_equal(df_result['manually_fixed_result_full'], [True, False])
        assert_array_equal(df_result['manually_fixed_result_partial'], [False, True])
