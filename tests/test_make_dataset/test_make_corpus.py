import os
import shutil

from unittest import mock

from src.datasets_manipulation.make_interim_dataset import make_corpus
from src.consts import WIKI_PT_TXT_FILE_NAME


class TestMakeCorpus(object):
    def test_make_corpus(self):
        test_data_dir = 'tests/test_make_dataset/mock_data'
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)
        os.makedirs(test_data_dir)
        assert len(os.listdir(test_data_dir)) == 0

        mock_WikiCorpus = mock.Mock()
        mock_WikiCorpus.return_value.get_texts.return_value = [['text1.1', 'text1.2'],
                                                               ['text2.1', 'text2.2'],
                                                               ['text3.1', 'text3.2'],
                                                               ['text4.1', 'text4.2'],
                                                               ['text5.1', 'text5.2']]
        with mock.patch('src.data.make_dataset.WikiCorpus', mock_WikiCorpus):
            make_corpus('', test_data_dir, size=2)

        assert len(os.listdir(test_data_dir)) == 3
        with open(test_data_dir + '/' + WIKI_PT_TXT_FILE_NAME + '00.txt') as f:
            assert f.read() == 'text1.1 text1.2\ntext2.1 text2.2\n'

    def test_make_corpus_without_split(self):
        test_data_dir = 'tests/test_make_dataset/mock_data'
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)
        os.makedirs(test_data_dir)
        assert len(os.listdir(test_data_dir)) == 0

        mock_WikiCorpus = mock.Mock()
        mock_WikiCorpus.return_value.get_texts.return_value = [['text1.1', 'text1.2'],
                                                               ['text2.1', 'text2.2'],
                                                               ['text3.1', 'text3.2'],
                                                               ['text4.1', 'text4.2'],
                                                               ['text5.1', 'text5.2']]
        with mock.patch('src.data.make_dataset.WikiCorpus', mock_WikiCorpus):
            make_corpus('', test_data_dir, split=False, size=2)

        assert len(os.listdir(test_data_dir)) == 1
        with open(test_data_dir + '/' + WIKI_PT_TXT_FILE_NAME + '00.txt') as f:
            assert f.read() == 'text1.1 text1.2\ntext2.1 text2.2\ntext3.1 text3.2\ntext4.1 text4.2\ntext5.1 text5.2\n'
