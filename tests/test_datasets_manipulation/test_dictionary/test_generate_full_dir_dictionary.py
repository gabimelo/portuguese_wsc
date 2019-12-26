import json
import os
from unittest import mock

from src.datasets_manipulation.dictionary import Dictionary

DIR_TEST_DATA = 'tests/test_datasets_manipulation/test_dictionary/mock_data/'


@mock.patch('src.datasets_manipulation.dictionary.consts.TEST_SET_FILE_NAME', DIR_TEST_DATA + 'test.txt')
@mock.patch('src.datasets_manipulation.dictionary.consts.VAL_SET_FILE_NAME', DIR_TEST_DATA + 'val.txt')
@mock.patch('src.datasets_manipulation.dictionary.consts.TRAIN_SET_FILE_NAME', DIR_TEST_DATA + 'train.txt')
@mock.patch('src.datasets_manipulation.dictionary.consts.FILTER_WORDS', 0)
@mock.patch('src.datasets_manipulation.dictionary.consts.CORPUS_DICTIONARY_FILE_NAME', DIR_TEST_DATA + 'dict.pkl')
@mock.patch(
    'src.datasets_manipulation.dictionary.consts.FILE_TOKEN_COUNT_DICT_FILE_NAME',
    DIR_TEST_DATA + 'file_token_count.json'
)
class TestGenerateFullDirDictionary():
    def test_generate_full_dir_dictionary(self):
        os.remove(DIR_TEST_DATA + 'dict.pkl')

        dictionary = Dictionary()
        dictionary.generate_full_dir_dictionary()

        expected_word2idx, expected_word_count, expected_idx2word = self.generate_expected_objects()

        assert dictionary.word2idx == expected_word2idx
        assert dictionary.word_count == expected_word_count
        assert dictionary.idx2word == expected_idx2word
        assert os.path.isfile(DIR_TEST_DATA + 'dict.pkl')

        with open(DIR_TEST_DATA + 'file_token_count.json', 'r') as f:
            file_token_count_json = json.loads(f.read())
        assert file_token_count_json[DIR_TEST_DATA + 'test.txt'] == 3
        assert file_token_count_json[DIR_TEST_DATA + 'val.txt'] == 6
        assert file_token_count_json[DIR_TEST_DATA + 'train.txt'] == 6

    @staticmethod
    def generate_expected_objects():
        expected_word2idx = {}
        expected_word_count = {}
        expected_idx2word = [
            '<unk>', 'text5.1', 'text5.2', '<eos>', 'text1.1', 'text1.2',
            'text2.1', 'text2.2', 'text3.1', 'text3.2', 'text4.1', 'text4.2'
        ]
        for i, item in enumerate(expected_idx2word):
            expected_word2idx[item] = i
            expected_word_count[item] = 1
        expected_word_count['<eos>'] = 5
        expected_word_count['<unk>'] = 0

        return expected_word2idx, expected_word_count, expected_idx2word
