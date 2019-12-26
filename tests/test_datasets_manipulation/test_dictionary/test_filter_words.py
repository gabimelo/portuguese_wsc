from src.datasets_manipulation.dictionary import Dictionary


class TestFilterWords():
    def test_filter_words(self):
        FILTER_WORDS = 2

        dic = Dictionary()
        dic.add_word('delete')
        dic.add_word('keep')
        dic.add_word('keep')
        dic.add_word('keep2')
        dic.add_word('delete2')
        dic.add_word('keep2')

        dic.filter_words(FILTER_WORDS)

        assert list(dic.word2idx.keys()) == ['<unk>', 'keep', 'keep2']
        assert list(dic.word2idx.values()) == [0, 1, 2]
        assert dic.word_count == {'<unk>': 2, 'keep': 2, 'keep2': 2}
        assert dic.idx2word == ['<unk>', 'keep', 'keep2']
