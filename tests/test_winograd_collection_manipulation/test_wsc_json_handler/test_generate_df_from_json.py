from unittest import mock

from numpy.testing import assert_array_equal

from src.winograd_collection_manipulation.wsc_json_handler import generate_df_from_json


class TestGenerateDfFromJson:
    @mock.patch(
        'src.winograd_collection_manipulation.wsc_json_handler.WINOGRAD_SCHEMAS_FILE',
        'data/processed/portuguese_wsc.json'
    )
    def test_portuguese_df(self):
        df = generate_df_from_json()

        assert_array_equal(df.columns.values,  [
            'correct_sentence', 'incorrect_sentence',
            'manually_fixed_correct_sentence', 'manually_fixed_incorrect_sentence',
            'correct_switched', 'incorrect_switched', 'is_switchable',
            'is_associative', 'translated'
        ])

        assert len(df) == 284
        assert df.translated.sum() == 277
        assert df.is_associative.sum() == 37
        assert df.is_switchable.sum() == 135

    @mock.patch(
        'src.winograd_collection_manipulation.wsc_json_handler.WINOGRAD_SCHEMAS_FILE',
        'data/processed/english_wsc.json'
    )
    def test_english_df(self):
        df = generate_df_from_json()
        assert_array_equal(df.columns.values,  [
            'correct_sentence', 'incorrect_sentence',
            'manually_fixed_correct_sentence', 'manually_fixed_incorrect_sentence',
            'correct_switched', 'incorrect_switched', 'is_switchable',
            'is_associative', 'translated'
        ])

        assert len(df) == 273
        assert df.translated.sum() == 273
        assert df.is_associative.sum() == 37
        assert df.is_switchable.sum() == 131
