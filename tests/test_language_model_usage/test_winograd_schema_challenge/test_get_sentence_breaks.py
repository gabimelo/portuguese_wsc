from src.language_model_usage.winograd_schema_challenge import get_sentence_breaks


class TestGetSentenceBreaks:
    def test_get_sentence_breaks(self):
        first_sentence = \
            'The city councilmen refused the demonstrators a permit because the city councilmen feared violence.'
        second_sentence = \
            'The city councilmen refused the demonstrators a permit because the demonstrators feared violence.'
        i = get_sentence_breaks(first_sentence, second_sentence)
        assert ' '.join(first_sentence.split()[:i]) == \
            'The city councilmen refused the demonstrators a permit because the'
        assert ' '.join(second_sentence.split()[:i]) == \
            'The city councilmen refused the demonstrators a permit because the'
        assert ' '.join(first_sentence.split()[i:]) == \
            'city councilmen feared violence.'
        assert ' '.join(second_sentence.split()[i:]) == \
            'demonstrators feared violence.'
        assert ' '.join(first_sentence.split()[:i]) == ' '.join(second_sentence.split()[:i])

        first_sentence = \
            'Os vereadores recusaram a autorização aos manifestantes porque os vereadores temiam a violência.'
        second_sentence = \
            'Os vereadores recusaram a autorização aos manifestantes porque os manifestantes temiam a violência.'

        i = get_sentence_breaks(first_sentence, second_sentence)
        assert ' '.join(first_sentence.split()[:i]) == \
            'Os vereadores recusaram a autorização aos manifestantes porque os'
        assert ' '.join(second_sentence.split()[:i]) == \
            'Os vereadores recusaram a autorização aos manifestantes porque os'
        assert ' '.join(first_sentence.split()[i:]) == \
            'vereadores temiam a violência.'
        assert ' '.join(second_sentence.split()[i:]) == \
            'manifestantes temiam a violência.'
        assert ' '.join(first_sentence.split()[:i]) == ' '.join(second_sentence.split()[:i])
