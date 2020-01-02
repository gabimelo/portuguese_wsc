import pytest

from src.language_model_usage.winograd_schema_challenge import get_partial_probs


@pytest.fixture()
def correct_sentence():
    correct_sentence = 'This is the correct sentence'

    return correct_sentence


@pytest.fixture()
def wrong_sentence():
    wrong_sentence = 'This is the wrong sentence'

    return wrong_sentence


class TestGetPartialProbs:
    def test_get_partial_probs(self, correct_sentence, wrong_sentence):
        correct_words_probs, wrong_words_probs = \
            get_partial_probs(
                correct_sentence, wrong_sentence,
                [1, 0.3, 0.2, 0.1, 0.7], [1, 0.3, 0.2, 0.1, 0.5]
            )

        assert correct_words_probs == [0.7]
        assert wrong_words_probs == [0.5]
