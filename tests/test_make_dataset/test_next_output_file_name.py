from src.wikidump import next_output_file_name
from src.consts import WIKI_PT_TXT_DIR_NAME, WIKI_PT_TXT_FILE_NAME


class TestNextOutputFileName(object):
    def test_next_output_file_name(self):
        assert next_output_file_name(WIKI_PT_TXT_DIR_NAME, 0) == \
            WIKI_PT_TXT_DIR_NAME + '/' + WIKI_PT_TXT_FILE_NAME + '00' + '.txt'
