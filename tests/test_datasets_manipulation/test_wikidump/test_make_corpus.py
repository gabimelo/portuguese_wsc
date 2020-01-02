import os
import shutil

from unittest import mock

from src.datasets_manipulation.wikidump import make_corpus_files
from src.helpers.consts import WIKI_PT_TXT_FILE_BASE_NAME


class TestMakeCorpus():
    def test_make_corpus(self):
        test_data_dir = 'tests/test_datasets_manipulation/test_wikidump/mock_data'
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)
        os.makedirs(test_data_dir)
        assert len(os.listdir(test_data_dir)) == 0

        mock_WikiCorpus = mock.Mock()
        mock_WikiCorpus.return_value.get_texts.return_value = [['text11', 'text12'],
                                                               ['text21', 'text22'],
                                                               ['text31', 'text32'],
                                                               ['text41', 'text42'],
                                                               ['text51', 'text52']]
        with mock.patch('src.datasets_manipulation.wikidump.WikiCorpus', mock_WikiCorpus):
            make_corpus_files('', test_data_dir, size=2)

        assert len(os.listdir(test_data_dir)) == 3
        with open(test_data_dir + '/' + WIKI_PT_TXT_FILE_BASE_NAME + '00.txt') as f:
            assert f.read() == 'text11 text12\ntext21 text22\n'
        with open(test_data_dir + '/' + WIKI_PT_TXT_FILE_BASE_NAME + '01.txt') as f:
            assert f.read() == 'text31 text32\ntext41 text42\n'
        with open(test_data_dir + '/' + WIKI_PT_TXT_FILE_BASE_NAME + '02.txt') as f:
            assert f.read() == 'text51 text52\n'

    def test_make_corpus_without_split(self):
        test_data_dir = 'tests/test_datasets_manipulation/test_wikidump/mock_data'
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)
        os.makedirs(test_data_dir)
        assert len(os.listdir(test_data_dir)) == 0

        mock_WikiCorpus = mock.Mock()
        mock_WikiCorpus.return_value.get_texts.return_value = [['text11', 'text12'],
                                                               ['text21', 'text22'],
                                                               ['text31', 'text32'],
                                                               ['text41', 'text42'],
                                                               ['text51', 'text52']]
        with mock.patch('src.datasets_manipulation.wikidump.WikiCorpus', mock_WikiCorpus):
            make_corpus_files('', test_data_dir, split=False, size=2)

        assert len(os.listdir(test_data_dir)) == 1
        with open(test_data_dir + '/' + WIKI_PT_TXT_FILE_BASE_NAME + '00.txt') as f:
            assert f.read() == 'text11 text12\ntext21 text22\ntext31 text32\ntext41 text42\ntext51 text52\n'
