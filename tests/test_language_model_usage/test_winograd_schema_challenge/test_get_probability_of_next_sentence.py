import pytest
from numpy.testing import assert_almost_equal
from transformers import BertTokenizer, BertForNextSentencePrediction

from src.language_model_usage.winograd_schema_challenge import get_probability_of_next_sentence


@pytest.mark.slow
class TestGetProbabilityOfNextSentence:
    def test_get_probability_of_next_sentence(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

        text1 = "How old are you?"
        text2 = "The Eiffel Tower is in Paris"
        text3 = "I am 22 years old"
        prob1 = get_probability_of_next_sentence(tokenizer, model, text1, text2)
        prob2 = get_probability_of_next_sentence(tokenizer, model, text1, text3)

        assert_almost_equal(prob1, 4.1673e-04)
        assert_almost_equal(prob2, 4.1673e-04)

    def test_get_probability_of_next_sentence_multilingual(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        model = BertForNextSentencePrediction.from_pretrained('bert-base-multilingual-uncased')

        text1 = "How old are you?"
        text2 = "The Eiffel Tower is in Paris"
        text3 = "I am 22 years old"
        prob1 = get_probability_of_next_sentence(tokenizer, model, text1, text2)
        prob2 = get_probability_of_next_sentence(tokenizer, model, text1, text3)

        assert_almost_equal(prob1, 4.1673e-04)
        assert_almost_equal(prob2, 4.1673e-04)

        text1 = "Quantos anos você tem?"
        text2 = "A Torre Eiffel fica em Paris"
        text3 = "Eu tenho 22 anos"
        prob = get_probability_of_next_sentence(tokenizer, model, text1, text2)

        assert_almost_equal(prob1, 4.1673e-04)
        assert_almost_equal(prob2, 4.1673e-04)

    def test_get_probability_of_next_sentence_portuguesee(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-portuguese-uncased')
        model = BertForNextSentencePrediction.from_pretrained('bert-base-portuguese-uncased')

        text1 = "Quantos anos você tem?"
        text2 = "A Torre Eiffel fica em Paris"
        text3 = "Eu tenho 22 anos"
        prob = get_probability_of_next_sentence(tokenizer, model, text1, text2)

        assert_almost_equal(prob1, 4.1673e-04)
        assert_almost_equal(prob2, 4.1673e-04)

