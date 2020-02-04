import pytest
from numpy.testing import assert_almost_equal
from transformers import BertTokenizer, BertForNextSentencePrediction

from src.language_model_usage.winograd_schema_challenge import get_probability_of_next_sentence


@pytest.mark.slow
class TestGetProbabilityOfNextSentence:
    def test_get_probability_of_next_sentence(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        model = BertForNextSentencePrediction.from_pretrained('bert-base-cased')

        text1 = "How old are you?"
        text2 = "The Eiffel Tower is in Paris"
        text3 = "I am 22 years old"
        prob1 = get_probability_of_next_sentence(tokenizer, model, text1, text2)
        prob2 = get_probability_of_next_sentence(tokenizer, model, text1, text3)

        assert_almost_equal(prob1, 0.0149559)
        assert_almost_equal(prob2, 0.9997911)

    def test_get_probability_of_next_sentence_multilingual(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertForNextSentencePrediction.from_pretrained('bert-base-multilingual-cased')

        text1 = "How old are you?"
        text2 = "The Eiffel Tower is in Paris"
        text3 = "I am 22 years old"
        prob1 = get_probability_of_next_sentence(tokenizer, model, text1, text2)
        prob2 = get_probability_of_next_sentence(tokenizer, model, text1, text3)

        assert_almost_equal(prob1, 0.5525756)
        assert_almost_equal(prob2, 0.9784408)

        text1 = "Quantos anos você tem?"
        text2 = "A Torre Eiffel fica em Paris"
        text3 = "Eu tenho 22 anos"
        prob1 = get_probability_of_next_sentence(tokenizer, model, text1, text2)
        prob2 = get_probability_of_next_sentence(tokenizer, model, text1, text3)

        assert_almost_equal(prob1, 0.8567284)
        assert_almost_equal(prob2, 0.9410717)

    def test_get_probability_of_next_sentence_portuguesee(self):
        tokenizer = BertTokenizer.from_pretrained('models/neuralmind/bert-base-portuguese-cased')
        model = BertForNextSentencePrediction.from_pretrained('models/neuralmind/bert-base-portuguese-cased')

        text1 = "Quantos anos você tem?"
        text2 = "A Torre Eiffel fica em Paris"
        text3 = "Eu tenho 22 anos"
        prob1 = get_probability_of_next_sentence(tokenizer, model, text1, text2)
        prob2 = get_probability_of_next_sentence(tokenizer, model, text1, text3)

        assert_almost_equal(prob1, 0.5229671)
        assert_almost_equal(prob2, 0.9979677)
