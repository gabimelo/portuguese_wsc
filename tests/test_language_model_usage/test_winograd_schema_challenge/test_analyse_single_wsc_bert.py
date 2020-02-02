import pytest
from numpy.testing import assert_almost_equal
from transformers import BertTokenizer, BertForNextSentencePrediction

from src.language_model_usage.winograd_schema_challenge import analyse_single_wsc_bert


@pytest.mark.slow
class TestGetProbabilityOfNextSentence:
    def test_get_probability_of_next_sentence(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

        wrong_sentence = "How old are you? The Eiffel Tower is in Paris"
        correct_sentence = "How old are you? I am 22 years old"

        full, partial = analyse_single_wsc_bert(model, tokenizer, correct_sentence, wrong_sentence)

        assert not full
        assert partial == 0

    def test_get_probability_of_next_sentence_multilingual(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        model = BertForNextSentencePrediction.from_pretrained('bert-base-multilingual-uncased')

        wrong_sentence = "How old are you? The Eiffel Tower is in Paris"
        correct_sentence = "How old are you? I am 22 years old"

        full, partial = analyse_single_wsc_bert(model, tokenizer, correct_sentence, wrong_sentence)

        assert not full
        assert partial == 0

        wrong_sentence = "Quantos anos você tem? A Torre Eiffel fica em Paris"
        correct_sentence = "Quantos anos você tem? Eu tenho 22 anos"

        full, partial = analyse_single_wsc_bert(model, tokenizer, correct_sentence, wrong_sentence)

        assert not full
        assert partial == 0

    def test_get_probability_of_next_sentence_portuguesee(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-portuguese-uncased')
        model = BertForNextSentencePrediction.from_pretrained('bert-base-portuguese-uncased')

        wrong_sentence = "Quantos anos você tem? A Torre Eiffel fica em Paris"
        correct_sentence = "Quantos anos você tem? Eu tenho 22 anos"

        full, partial = analyse_single_wsc_bert(model, tokenizer, correct_sentence, wrong_sentence)

        assert not full
        assert partial == 0

