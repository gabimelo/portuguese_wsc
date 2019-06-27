from unittest import mock

from src.helpers.utils import get_latest_model_file


class TestGetLatestFile(object):
    @mock.patch('src.helpers.utils.MODEL_FILE_NAME', 'tests/test_helpers/test_utils/mock_dir/{}.pt')
    def test_get_latest_file(self):
        assert get_latest_model_file() == 'tests/test_helpers/test_utils/mock_dir/newer_file.pt'
