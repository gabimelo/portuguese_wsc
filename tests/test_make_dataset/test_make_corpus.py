from src.data.make_dataset import make_corpus
from src.consts import WIKI_PT_TXT_DIR_NAME, WIKI_PT_TXT_FILE_NAME


class TestMakeCorpus(object):
    def test_make_corpus(self):
        assert make_corpus(WIKI_PT_TXT_DIR_NAME, 0) == \
            WIKI_PT_TXT_DIR_NAME + '/' + WIKI_PT_TXT_FILE_NAME + '00' + '.txt'
