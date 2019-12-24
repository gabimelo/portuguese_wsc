import pandas as pd
from bs4 import BeautifulSoup

from src.helpers.consts import WINOGRAD_PT_HTML_SCHEMAS_FILE, MISSING_TRANSLATION_INDEXES


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


def generate_df_from_html():
    with open(WINOGRAD_PT_HTML_SCHEMAS_FILE, 'r') as f:
        soup = BeautifulSoup(f, 'html5lib')

    rows = []
    for item in soup.find('ol').find_all('li', recursive=False):
        pronoun = item.find('b').text.strip()

        content = join_content(item)
        correct_answer = content[-1].replace('Resposta Correta:', '').strip()[0]
        substitution_a = content[-3]
        substitution_b = content[-2]

        schema, snippet = get_schema_and_snippet_texts(item)

        translated = True

        row = [schema, snippet, pronoun, correct_answer, substitution_a, substitution_b, translated]
        rows.append(row)

    df = pd.DataFrame(rows, columns=['schema', 'snippet', 'pronoun', 'correct_answer',
                                     'substitution_a', 'substitution_b', 'translated'])

    for index in MISSING_TRANSLATION_INDEXES:
        df.loc[index, 'translated'] = False

    return df
