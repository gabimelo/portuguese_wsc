from src.datasets_manipulation.wikidump import get_output_file_name
from src.helpers.consts import WIKI_PT_TXT_DIR_NAME, WIKI_PT_TXT_FILE_BASE_NAME


class TestNextOutputFileName():
    def test_get_output_file_name(self):
        assert get_output_file_name(WIKI_PT_TXT_DIR_NAME, 0) == \
            WIKI_PT_TXT_DIR_NAME + '/' + WIKI_PT_TXT_FILE_BASE_NAME + '00' + '.txt'
