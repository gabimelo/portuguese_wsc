import pickle
from unittest import mock

import pytest

from src.language_model_usage.winograd_schema_challenge import analyse_single_wsc


def mock_generate(model_file_name, corpus, device, input_wsc):
    possible_values = {
        28: [1, 0.3, 0.2, 0.1, 0.7],  # what gets returned when called with correct_sentence
        26: [1, 0.3, 0.2, 0.1, 0.5]  # what gets returned when called with wrong_sentence
    }
    print(f'passing through here with {input_wsc}')

    return None, possible_values[len(input_wsc)]


@pytest.fixture()
def corpus():
    corpus_file_name = 'tests/test_language_model_usage/mock_data/corpus.pkl'
    corpus = pickle.load(open(corpus_file_name, "rb"))

    return corpus


@pytest.fixture()
def correct_sentence():
    correct_sentence = 'This is the correct sentence'

    return correct_sentence


@pytest.fixture()
def wrong_sentence():
    wrong_sentence = 'This is the wrong sentence'

    return wrong_sentence


@mock.patch('src.language_model_usage.winograd_schema_challenge.generate', side_effect=mock_generate)
class TestAnalyseSingleWsc:
    def test_correct(self, mocked_generate, correct_sentence, wrong_sentence, corpus):
        result = analyse_single_wsc(
            'dummy_model_file_name', corpus, 'mock_device', correct_sentence, wrong_sentence, partial=False
        )

        assert result

    def test_same_sentences(self, mocked_generate, correct_sentence, wrong_sentence, corpus):
        result = analyse_single_wsc(
            'dummy_model_file_name', corpus, 'mock_device', correct_sentence, correct_sentence, partial=False
        )

        assert not result

    def test_inverted_sentences(self, mocked_generate, correct_sentence, wrong_sentence, corpus):
        result = analyse_single_wsc(
            'dummy_model_file_name', corpus, 'mock_device', wrong_sentence, correct_sentence, partial=False
        )

        assert not result

    def test_partial(self, mocked_generate, correct_sentence, wrong_sentence, corpus):
        result = analyse_single_wsc(
            'dummy_model_file_name', corpus, 'mock_device', correct_sentence, wrong_sentence, partial=True
        )

        assert result

    def test_partial_same_sentences(self, mocked_generate, correct_sentence, wrong_sentence, corpus):
        result = analyse_single_wsc(
            'dummy_model_file_name', corpus, 'mock_device', correct_sentence, correct_sentence, partial=True
        )

        assert not result

    def test_partial_inverted_sentences(self, mocked_generate, correct_sentence, wrong_sentence, corpus):
        result = analyse_single_wsc(
            'dummy_model_file_name', corpus, 'mock_device', wrong_sentence, correct_sentence, partial=True
        )

        assert not result
