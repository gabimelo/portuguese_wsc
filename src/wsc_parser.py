import json
import re

import pandas as pd
from bs4 import BeautifulSoup

from src.consts import WINOGRAD_SCHEMAS_FILE


def join_content(item):
    content = [it.strip().replace('  ', ' ') for it in item.text.split('\n') if it.strip() != '']
    return content


def clean_tags_for_schema_and_snippet(item):
    fonts = item.find_all('font')
    for font in fonts:
        if font is not None:
            font.extract()
    ol = item.find('ol')
    ol.extract()
    ps = item.find_all('p')
    for p in ps:
        p.extract()

    return item


def get_schema_and_snippet_texts(item):
    item = clean_tags_for_schema_and_snippet(item)
    content = join_content(item)

    texts = (' ').join(content).split('Trecho:')
    schema = texts[0].replace('  ', ' ').strip()
    snippet = texts[1].replace('  ', ' ').strip()
    if not schema[-1] in ['.', '!', '?']:
        schema += '.'
    if snippet[-1] != schema[-1] and schema[:-1].split()[-1] == snippet.split()[-1]:
        snippet += schema[-1]

    return schema, snippet


def generate_df(full_english_text):
    with open('data/processed/port_wsc.html', 'r') as f:
        soup = BeautifulSoup(f, 'html5lib')

    rows = []
    for item in soup.find('ol').find_all('li', recursive=False):
        pronoun = item.find('b').text.strip()

        content = join_content(item)
        correct_answer = content[-1].replace('Resposta Correta:', '').strip()[0]
        substitution_a = content[-3]
        substitution_b = content[-2]

        schema, snippet = get_schema_and_snippet_texts(item)

        translated = False if schema[:20] in full_english_text else True

        row = [schema, snippet, pronoun, correct_answer, substitution_a, substitution_b, translated]
        rows.append(row)

    df = pd.DataFrame(rows, columns=['schema', 'snippet', 'pronoun', 'correct_answer',
                                     'substitution_a', 'substitution_b', 'translated'])
    return df


def generate_df_from_json():
    rows = []
    with open(WINOGRAD_SCHEMAS_FILE, 'r', encoding='utf-8') as fp:
        wsc_json = json.load(fp)

    if 'substitution' in wsc_json[0]:
        for i in range(0, len(wsc_json), 2):
            correct_sentence = wsc_json[i]['substitution'] if wsc_json[i]['correctness'] \
                else wsc_json[i+1]['substitution']  # noqa E226
            incorrect_sentence = wsc_json[i]['substitution'] if not wsc_json[i]['correctness'] \
                else wsc_json[i+1]['substitution'] # noqa E226
            rows.append([correct_sentence, incorrect_sentence])
        df = pd.DataFrame(rows, columns=['correct_sentence', 'incorrect_sentence'])
    else:
        for i in range(wsc_json):
            rows.append([wsc_json[i]['correct_sentence'], wsc_json[i]['incorrect_sentence'],
                         wsc_json[i]['manually_fixed_correct_sentence'],
                         wsc_json[i]['manually_fixed_incorrect_sentence'],
                         wsc_json[i]['correct_switched'], wsc_json[i]['incorrect_switched'],
                         wsc_json[i]['manually_fixed_correct_switched'],
                         wsc_json[i]['manually_fixed_incorrect_switched'],
                         wsc_json[i]['is_switchable'], wsc_json[i]['is_associative'],
                         wsc_json[i]['translated']])

        df = pd.DataFrame(rows, columns=['correct_sentence', 'incorrect_sentence',
                                         'manually_fixed_correct_sentence', 'manually_fixed_incorrect_sentence',
                                         'correct_switched', 'incorrect_switched',
                                         'manually_fixed_correct_switched', 'manually_fixed_incorrect_switched',
                                         'is_switchable', 'is_associative', 'translated'])

    return df


def generate_json(df):
    json_rows = []
    for index, row in df.iterrows():
        dic = {'question_id': index,
               'correct_sentence': row.correct_sentence,
               'correct_switched': row.correct_switched,
               'manually_fixed_correct_sentence': row.manually_fixed_correct_sentence,
               'manually_fixed_correct_switched': row.manually_fixed_correct_switched,
               'incorrect_sentence': row.incorrect_sentence,
               'incorrect_switched': row.incorrect_switched,
               'manually_fixed_incorrect_sentence': row.manually_fixed_incorrect_sentence,
               'manually_fixed_incorrect_switched': row.manually_fixed_incorrect_switched}

        dic['is_associative'] = False if 'is_associative' not in row else row.is_associative
        dic['is_switchable'] = False if 'is_switchable' not in row else row.is_switchable
        dic['translated'] = row.translated

        json_rows.append(dic)

    with open('data/processed/portuguese_wsc.json', 'w') as outfile:
        json.dump(json_rows, outfile)


def generate_full_sentences(row):
    if row.pronoun.islower() and (row.substitution_b.lower() in row.schema or
                                  row.substitution_b.lower() not in row.schema.lower()):
        subs_a = row.substitution_a.lower()
        subs_b = row.substitution_b.lower()
    else:
        subs_a = row.substitution_a
        subs_b = row.substitution_b

    new_snippet_a = re.sub(r"\b%s\b" % row.pronoun, subs_a, row.snippet)
    new_snippet_b = re.sub(r"\b%s\b" % row.pronoun, subs_b, row.snippet)
    new_schema_a = row.schema.replace(row.snippet, new_snippet_a).strip()
    new_schema_b = row.schema.replace(row.snippet, new_snippet_b).strip()
    new_switched_a = row.switched.replace(row.snippet, new_snippet_a).strip()
    new_switched_b = row.switched.replace(row.snippet, new_snippet_b).strip()

    if row.correct_answer.lower() == 'a':
        return new_schema_a, new_schema_b, new_switched_b, new_switched_a
    return new_schema_b, new_schema_a, new_switched_a, new_switched_b
