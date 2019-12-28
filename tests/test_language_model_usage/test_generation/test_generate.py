import pickle
from unittest import mock

import numpy as np
import pytest
import torch
from numpy.testing import assert_array_almost_equal

from src.language_model_usage.generation import generate


@pytest.fixture()
def corpus():
    corpus_file_name = 'tests/test_language_model_usage/mock_data/corpus.pkl'
    corpus = pickle.load(open(corpus_file_name, "rb"))

    return corpus


@pytest.fixture()
def dummy_model_file_name():
    return 'tests/test_language_model_usage/mock_data/dummy_model_file_name.pt'


def mocked_model_return(*args):
    corpus_dictionary_length = 12
    output = torch.Tensor([list(np.arange(0, corpus_dictionary_length))])

    return output, None


class TestGenerate():
    def test_wsc_generation(self, corpus, dummy_model_file_name):
        input_wsc = 'text51 text52 text11 text32'
        input_words, input_words_probs, mocked_multinominal = self.base_test(corpus, dummy_model_file_name, input_wsc)

        assert input_words == ['text51', 'text52', 'text11', 'text32']
        assert len(input_words_probs) == len(input_words)
        assert_array_almost_equal(input_words_probs, [0.0666667, 7.8010e-05, 5.7642e-04, 8.5549e-02], decimal=6)
        assert not mocked_multinominal.called

    @mock.patch('src.language_model_usage.generation.WORDS_TO_GENERATE', 3)
    def test_non_wsc_generation(self, corpus, dummy_model_file_name):
        input_words, input_words_probs, mocked_multinominal = self.base_test(corpus, dummy_model_file_name)

        assert len(input_words_probs) == 3
        assert len(input_words) == 3
        assert mocked_multinominal.call_count == 3

    @staticmethod
    def base_test(corpus, dummy_model_file_name, input_wsc=None):
        device = 'cpu'

        mocked_model = mock.Mock()
        mocked_model.to.return_value = mock.Mock(side_effect=mocked_model_return)

        mocked_multinominal = mock.Mock(side_effect=torch.multinomial)

        with mock.patch('src.language_model_usage.generation.torch.load', return_value=mocked_model), \
           mock.patch('src.language_model_usage.generation.torch.multinomial', mocked_multinominal):  # noqa: E127
            input_words, input_words_probs = generate(dummy_model_file_name, corpus, device, input_wsc)

        return input_words, input_words_probs, mocked_multinominal
