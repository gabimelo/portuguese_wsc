from src.language_model_usage.winograd_schema_challenge import get_sentence_breaks


class TestGetSentenceBreaks:
    def test_get_sentence_breaks(self):
        first_sentence = \
            'The city councilmen refused the demonstrators a permit because the city councilmen feared violence.'
        second_sentence = \
            'The city councilmen refused the demonstrators a permit because the demonstrators feared violence.'
        correct_sentence_first_part, correct_sentence_second_part, wrong_sentence_first_part, wrong_sentence_second_part = \
            get_sentence_breaks(first_sentence, second_sentence)

        assert correct_sentence_first_part == 'The city councilmen refused the demonstrators a permit because the'
        assert correct_sentence_first_part == wrong_sentence_first_part
        assert correct_sentence_second_part == 'city councilmen feared violence.'
        assert wrong_sentence_second_part == 'demonstrators feared violence.'

        first_sentence = \
            'Os vereadores recusaram a autorização aos manifestantes porque os vereadores temiam a violência.'
        second_sentence = \
            'Os vereadores recusaram a autorização aos manifestantes porque os manifestantes temiam a violência.'
        correct_sentence_first_part, correct_sentence_second_part, wrong_sentence_first_part, wrong_sentence_second_part = \
            get_sentence_breaks(first_sentence, second_sentence)

        assert correct_sentence_first_part == 'Os vereadores recusaram a autorização aos manifestantes porque os'
        assert correct_sentence_first_part == wrong_sentence_first_part
        assert correct_sentence_second_part == 'vereadores temiam a violência.'
        assert wrong_sentence_second_part == 'manifestantes temiam a violência.'
