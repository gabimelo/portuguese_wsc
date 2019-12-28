import pandas as pd
import pytest

from src.language_model_usage.winograd_schema_challenge import (
    calculate_single_accuracy, calculate_accuracy, calculate_consistency, calculate_metrics
)


@pytest.fixture()
def df():
    df = pd.DataFrame({
        'is_associative':                [True,  True,   True,  True,  True,  False, False],
        'is_switchable':                 [True,  True,   False, False, True,  False, True],
        'original_result_full':          [False, True,   False, True,  False, True,  False],
        'original_result_partial':       [False, True,   True,  False, True,  False, True],
        'switched_result_full':          [False, False,  False, True,  False, True,  False],
        'switched_result_partial':       [False, False,  True,  False, True,  True,  False],
        'manually_fixed_result_full':    [False, False,  False, True,  False, True,  False],
        'manually_fixed_result_partial': [False, False,  True,  False, True,  False, True],
    })

    return df


class TestMetrics:
    @staticmethod
    def assert_metric_keys(metrics_dict, accuracy=True):
        if accuracy:
            expected_keys = ['accuracy_full', 'accuracy_partial', 'number_of_examples']
        else:
            expected_keys = ['consistency_full', 'consistency_partial', 'number_of_examples']

        assert set(metrics_dict.keys()) == set(expected_keys)

    @staticmethod
    def assert_metrics_values(metrics_dict, full, partial, examples, accuracy=True):
        if accuracy:
            full_key = 'accuracy_full'
            partial_key = 'accuracy_partial'
        else:
            full_key = 'consistency_full'
            partial_key = 'consistency_partial'
        assert metrics_dict[full_key] == full / examples
        assert metrics_dict[partial_key] == partial /examples
        assert metrics_dict['number_of_examples'] == examples

    def test_calculate_single_accuracy(self, df):
        actual_accuracy = calculate_single_accuracy(df, 'original_result_full')

        assert actual_accuracy == 3/7

    def test_calculate_accuracy(self, df):
        actual_acc_dict = calculate_accuracy(df, '', 'original')

        expected_number_of_examples = 7

        self.assert_metric_keys(actual_acc_dict)
        self.assert_metrics_values(actual_acc_dict, 3, 4, expected_number_of_examples)

    def test_calculate_accuracy_subset(self, df):
        actual_acc_dict = calculate_accuracy(df, 'is_switchable', 'original')

        expected_number_of_examples = 4

        self.assert_metric_keys(actual_acc_dict)
        self.assert_metrics_values(actual_acc_dict, 1, 3, expected_number_of_examples)

    def test_calculate_accuracy_non_subset(self, df):
        actual_acc_dict = calculate_accuracy(df, '~is_switchable', 'original')

        expected_number_of_examples = 3

        self.assert_metric_keys(actual_acc_dict)
        self.assert_metrics_values(actual_acc_dict, 2, 1, expected_number_of_examples)

    def test_calculate_consistency(self, df):
        actual_acc_dict = calculate_consistency(df)

        expected_number_of_examples = 4

        self.assert_metric_keys(actual_acc_dict, accuracy=False)
        self.assert_metrics_values(actual_acc_dict, 3, 2, expected_number_of_examples, accuracy=False)

    def test_metrics(self, df):
        actual_metrics = calculate_metrics(df, test_on_manually_fixed=True)

        assert set(actual_metrics.keys()) == set([
            'complete', 'associative', 'non_associative', 'switched', 'unswitched', 'consistency',
            'manually_fixed_complete', 'manually_fixed_associative', 'manually_fixed_non_associative'
        ])
        self.assert_metric_keys(actual_metrics['complete'])
        self.assert_metrics_values(actual_metrics['complete'], 3, 4, 7)
        self.assert_metric_keys(actual_metrics['associative'])
        self.assert_metrics_values(actual_metrics['associative'], 2, 3, 5)
        self.assert_metric_keys(actual_metrics['non_associative'])
        self.assert_metrics_values(actual_metrics['non_associative'], 1, 1, 2)
        self.assert_metric_keys(actual_metrics['switched'])
        self.assert_metrics_values(actual_metrics['switched'], 0, 1, 4)
        self.assert_metric_keys(actual_metrics['unswitched'])
        self.assert_metrics_values(actual_metrics['unswitched'], 1, 3, 4)
        self.assert_metric_keys(actual_metrics['consistency'], accuracy=False)
        self.assert_metrics_values(actual_metrics['consistency'], 3, 2, 4, accuracy=False)
        self.assert_metric_keys(actual_metrics['manually_fixed_complete'])
        self.assert_metrics_values(actual_metrics['manually_fixed_complete'], 2, 3, 7)
        self.assert_metric_keys(actual_metrics['manually_fixed_associative'])
        self.assert_metrics_values(actual_metrics['manually_fixed_associative'], 1, 2, 5)
        self.assert_metric_keys(actual_metrics['manually_fixed_non_associative'])
        self.assert_metrics_values(actual_metrics['manually_fixed_non_associative'], 1, 1, 2)


    def test_metrics_without_manually_fixed(self, df):
        actual_metrics = calculate_metrics(df, test_on_manually_fixed=False)

        assert set(actual_metrics.keys()) == set([
            'complete', 'associative', 'non_associative', 'switched', 'unswitched', 'consistency'
        ])
