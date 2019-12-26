import os
from unittest import mock

from src.datasets_manipulation.corpus import Corpus

DIR_TEST_DATA = 'tests/test_datasets_manipulation/test_my_tests/mock_data/'


@mock.patch('src.datasets_manipulation.dictionary.consts.TEST_SET_FILE_NAME', DIR_TEST_DATA + 'test.txt')
@mock.patch('src.datasets_manipulation.dictionary.consts.VAL_SET_FILE_NAME', DIR_TEST_DATA + 'val.txt')
@mock.patch('src.datasets_manipulation.dictionary.consts.TRAIN_SET_FILE_NAME', DIR_TEST_DATA + 'train.txt')
@mock.patch('src.datasets_manipulation.dictionary.consts.FILTER_WORDS', 0)
@mock.patch(
    'src.datasets_manipulation.dictionary.consts.FILE_TOKEN_COUNT_DICT_FILE_NAME',
    DIR_TEST_DATA + 'file_token_count.json'
)
@mock.patch(
    'src.datasets_manipulation.corpus.FILE_TOKEN_COUNT_DICT_FILE_NAME',
    DIR_TEST_DATA + 'file_token_count.json'
)
class TestCorpus():
    @mock.patch('src.datasets_manipulation.corpus.CORPUS_DICTIONARY_FILE_NAME', DIR_TEST_DATA + 'another_dict.pkl')
    @mock.patch(
        'src.datasets_manipulation.dictionary.consts.CORPUS_DICTIONARY_FILE_NAME',
        DIR_TEST_DATA + 'another_dict.pkl'
    )
    def test_initialize_corpus(self):
        if os.path.isfile(DIR_TEST_DATA + 'another_dict.pkl'):
            os.remove(DIR_TEST_DATA + 'another_dict.pkl')

        corpus = Corpus()
        assert corpus.train is None
        assert corpus.valid is None
        assert corpus.test is None
        assert os.path.isfile(DIR_TEST_DATA + 'another_dict.pkl')

        os.remove(DIR_TEST_DATA + 'another_dict.pkl')

    @mock.patch('src.datasets_manipulation.corpus.CORPUS_DICTIONARY_FILE_NAME', DIR_TEST_DATA + 'dict.pkl')
    def test_initialize_corpus_from_existing_dictionary(self):
        corpus = Corpus()
        assert corpus.train is None
        assert corpus.valid is None
        assert corpus.test is None
        assert type(corpus.dictionary).__name__ == 'Dictionary'
