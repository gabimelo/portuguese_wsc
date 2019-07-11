import json

import pandas as pd

from src.consts import WINOGRAD_SCHEMAS_FILE, WINOGRAD_SCHEMAS_ORIGINAL_FILE


def generate_df_from_original_json():
    with open(WINOGRAD_SCHEMAS_ORIGINAL_FILE, 'r', encoding='utf-8') as fp:
        wsc_json = json.load(fp)

    rows = []
    for i in range(0, len(wsc_json), 2):
        correct_sentence = wsc_json[i]['substitution'] if wsc_json[i]['correctness'] \
            else wsc_json[i+1]['substitution']  # noqa E226
        incorrect_sentence = wsc_json[i]['substitution'] if not wsc_json[i]['correctness'] \
            else wsc_json[i+1]['substitution'] # noqa E226
        correct_sentence = correct_sentence.replace('recieved', 'received')
        incorrect_sentence = incorrect_sentence.replace('recieved', 'received')
        rows.append([correct_sentence, incorrect_sentence])

    df = pd.DataFrame(rows, columns=['correct_sentence', 'incorrect_sentence'])

    return df


def generate_df_from_json():
    with open(WINOGRAD_SCHEMAS_FILE, 'r', encoding='utf-8') as fp:
        wsc_json = json.load(fp)

    rows = []
    for i in range(len(wsc_json)):
        rows.append([wsc_json[i]['correct_sentence'], wsc_json[i]['incorrect_sentence'],
                     wsc_json[i]['manually_fixed_correct_sentence'],
                     wsc_json[i]['manually_fixed_incorrect_sentence'],
                     wsc_json[i]['correct_switched'], wsc_json[i]['incorrect_switched'],
                     wsc_json[i]['is_switchable'], wsc_json[i]['is_associative'],
                     wsc_json[i]['translated']])

    df = pd.DataFrame(rows, columns=['correct_sentence', 'incorrect_sentence',
                                     'manually_fixed_correct_sentence', 'manually_fixed_incorrect_sentence',
                                     'correct_switched', 'incorrect_switched',
                                     'is_switchable', 'is_associative', 'translated'])

    return df


def generate_json(df):
    json_rows = []
    for index, row in df.iterrows():
        dic = {'question_id': index,
               'correct_sentence': row.correct_sentence,
               'incorrect_sentence': row.incorrect_sentence,
               'manually_fixed_correct_sentence': row.manually_fixed_correct_sentence,
               'manually_fixed_incorrect_sentence': row.manually_fixed_incorrect_sentence,
               'correct_switched': row.manually_fixed_correct_switched,
               'incorrect_switched': row.manually_fixed_incorrect_switched}

        dic['is_associative'] = False if 'is_associative' not in row else row.is_associative
        dic['is_switchable'] = False if 'is_switchable' not in row else row.is_switchable
        if dic['is_switchable'] and dic['correct_switched'] == '':
            dic['correct_switched'] = row.correct_switched
            dic['incorrect_switched'] = row.incorrect_switched
        dic['translated'] = row.translated

        json_rows.append(dic)

    with open(WINOGRAD_SCHEMAS_FILE, 'w') as outfile:
        json.dump(json_rows, outfile, ensure_ascii=False, indent=2)
