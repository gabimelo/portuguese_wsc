import json

import pandas as pd
from bs4 import BeautifulSoup


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


# TODO these two parameters and code related to them must be removed
def generate_df(still_in_english, subs_not_working):
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

        skip = False
        for english_sentence in still_in_english:
            if english_sentence[:20] in schema:
                skip = True
        if skip:
            continue
        for not_working_sentence in subs_not_working:
            if not_working_sentence[:20] in schema:
                skip = True
        if skip:
            continue

        row = [schema, snippet, pronoun, correct_answer, substitution_a, substitution_b]
        rows.append(row)

    df = pd.DataFrame(rows, columns=['schema', 'snippet', 'pronoun', 'correct_answer',
                                     'substitution_a', 'substitution_b'])
    return df


def generate_df_from_json():
    rows = []
    with open('./data/processed/english_wsc.json', 'r', encoding='utf-8') as fp:
        wsc_json = json.load(fp)

    for i in range(0, len(wsc_json), 2):
        correct_sentence = wsc_json[i]['substitution'] if wsc_json[i]['correctness'] \
            else wsc_json[i+1]['substitution']  # noqa E226
        incorrect_sentence = wsc_json[i]['substitution'] if not wsc_json[i]['correctness'] \
            else wsc_json[i+1]['substitution'] # noqa E226
        rows.append([correct_sentence, incorrect_sentence])

    df = pd.DataFrame(rows, columns=['correct_sentence', 'incorrect_sentence'])

    return df