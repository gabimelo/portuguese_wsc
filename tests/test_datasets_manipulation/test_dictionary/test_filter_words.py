from unittest import mock

from src.datasets_manipulation.dictionary import Dictionary


class TestFilterWords(object):
    @mock.patch('src.datasets_manipulation.dictionary.MIN_APPEARANCES_FOR_WORD_IN_VOCAB', 2)
    def test_filter_words(self):
        dic = Dictionary()
        dic.add_word('delete')
        dic.add_word('keep')
        dic.add_word('keep')
        dic.add_word('keep2')
        dic.add_word('delete2')
        dic.add_word('keep2')

        dic.filter_words()

        assert list(dic.word2idx.keys()) == ['<unk>', 'keep', 'keep2']
        assert list(dic.word2idx.values()) == [0, 1, 2]
        assert hasattr(dic, 'word_count')
        assert dic.idx2word == ['<unk>', 'keep', 'keep2']
