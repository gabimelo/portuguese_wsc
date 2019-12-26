import os
from unittest import mock

from numpy.testing import assert_array_equal

from src.datasets_manipulation.corpus import Corpus

DIR_TEST_DATA = 'tests/test_datasets_manipulation/test_u_corpus/mock_data/'


@mock.patch(
    'src.datasets_manipulation.corpus.FILE_TOKEN_COUNT_DICT_FILE_NAME',
    DIR_TEST_DATA + 'file_token_count.json'
)
@mock.patch('src.datasets_manipulation.corpus.CORPUS_FILE_NAME', DIR_TEST_DATA + 'corpus.pkl')
@mock.patch('src.datasets_manipulation.corpus.CORPUS_DICTIONARY_FILE_NAME', DIR_TEST_DATA + 'dict.pkl')
@mock.patch('src.datasets_manipulation.corpus.TEST_SET_FILE_NAME', DIR_TEST_DATA + 'test.txt')
@mock.patch('src.datasets_manipulation.corpus.VAL_SET_FILE_NAME', DIR_TEST_DATA + 'val.txt')
@mock.patch('src.datasets_manipulation.corpus.TRAIN_SET_FILE_NAME', DIR_TEST_DATA + 'train.txt')
class TestAddCorpusData():
    def test_add_corpus_data(self):
        if os.path.isfile(DIR_TEST_DATA + 'corpus.pkl'):
            os.remove(DIR_TEST_DATA + 'corpus.pkl')

        corpus = Corpus()
        corpus.add_corpus_data()

        assert_array_equal(corpus.train.numpy(), [4, 5, 3, 6, 7, 3])
        assert_array_equal(corpus.test, [1, 2, 3])
        assert_array_equal(corpus.valid, [8, 9, 3, 10, 11, 3])

        assert os.path.isfile(DIR_TEST_DATA + 'corpus.pkl')