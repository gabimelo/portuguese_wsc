from unittest import mock

from src.dictionary import Dictionary


class TestFilterWords(object):
    @mock.patch('src.dictionary.MIN_APPEARANCES_FOR_WORD_IN_VOCAB', 2)
    def test_filter_words(self):
        dic = Dictionary()
        dic.add_word('deletar')
        dic.add_word('manter')
        dic.add_word('manter')
        dic.add_word('manter2')
        dic.add_word('deletar2')
        dic.add_word('manter2')

        dic.filter_words()

        assert list(dic.word2idx.keys()) == ['<UNK>', 'manter', 'manter2']
        assert list(dic.word2idx.values()) == [0, 1, 2]
        assert not hasattr(dic, 'word_count')
        assert dic.idx2word == ['<UNK>', 'manter', 'manter2']
