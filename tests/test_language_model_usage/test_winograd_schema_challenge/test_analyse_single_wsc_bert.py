import pytest
from transformers import BertTokenizer, BertForNextSentencePrediction

from src.language_model_usage.winograd_schema_challenge import analyse_single_wsc_bert


@pytest.mark.slow
class TestGetProbabilityOfNextSentence:
    def test_get_probability_of_next_sentence(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        model = BertForNextSentencePrediction.from_pretrained('bert-base-cased')

        wrong_sentence = "How old are you? The Eiffel Tower is in Paris"
        correct_sentence = "How old are you? I am 22 years old"

        full, partial = analyse_single_wsc_bert(model, tokenizer, correct_sentence, wrong_sentence)

        assert full
        assert partial == 0

        full, partial = analyse_single_wsc_bert(model, tokenizer, wrong_sentence, correct_sentence)

        assert not full
        assert partial == 0

    def test_get_probability_of_next_sentence_multilingual(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertForNextSentencePrediction.from_pretrained('bert-base-multilingual-cased')

        wrong_sentence = "How old are you? The Eiffel Tower is in Paris"
        correct_sentence = "How old are you? I am 22 years old"

        full, partial = analyse_single_wsc_bert(model, tokenizer, correct_sentence, wrong_sentence)

        assert full
        assert partial == 0

        full, partial = analyse_single_wsc_bert(model, tokenizer, wrong_sentence, correct_sentence)

        assert not full
        assert partial == 0

        wrong_sentence = "Quantos anos você tem? A Torre Eiffel fica em Paris"
        correct_sentence = "Quantos anos você tem? Eu tenho 22 anos"

        full, partial = analyse_single_wsc_bert(model, tokenizer, correct_sentence, wrong_sentence)

        assert full
        assert partial == 0

        full, partial = analyse_single_wsc_bert(model, tokenizer, wrong_sentence, correct_sentence)

        assert not full
        assert partial == 0

    def test_get_probability_of_next_sentence_portuguesee(self):
        tokenizer = BertTokenizer.from_pretrained('models/neuralmind/bert-base-portuguese-cased')
        model = BertForNextSentencePrediction.from_pretrained('models/neuralmind/bert-base-portuguese-cased')

        wrong_sentence = "Quantos anos você tem? A Torre Eiffel fica em Paris"
        correct_sentence = "Quantos anos você tem? Eu tenho 22 anos"

        full, partial = analyse_single_wsc_bert(model, tokenizer, correct_sentence, wrong_sentence)

        assert full
        assert partial == 0

        full, partial = analyse_single_wsc_bert(model, tokenizer, wrong_sentence, correct_sentence)

        assert not full
        assert partial == 0
