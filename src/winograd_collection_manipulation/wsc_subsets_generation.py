import re
import json

from consts import (
    WINOGRAD_ASSOCIATIVE_LABEL_FILE, WINOGRAD_SWITCHED_LABEL_FILE, CAPITALIZED_WORD_LIST_FILE,
    PORTUGUESE, MANUAL_PT_FIXES_FILE
)
from src.winograd_collection_manipulation.wsc_html_parser import generate_df
from src.winograd_collection_manipulation.wsc_json_parser import generate_df_from_json, generate_json
from src.winograd_collection_manipulation.text_manipulation import custom_tokenizer


def generate_full_sentences(row):
    subs_a = row.substitution_a.lower()
    subs_b = row.substitution_b.lower()

    new_snippet_a = re.sub(r"\b%s\b" % row.pronoun, subs_a, row.snippet)
    new_snippet_b = re.sub(r"\b%s\b" % row.pronoun, subs_b, row.snippet)

    if new_snippet_a[0] == '[' and new_snippet_a[-1] == ']':
        new_snippet_a = new_snippet_a[1:-1]
    if new_snippet_b[0] == '[' and new_snippet_b[-1] == ']':
        new_snippet_b = new_snippet_b[1:-1]

    if 'schema' in row.index:
        new_schema_a = row.schema.replace(row.snippet, new_snippet_a).strip()
        new_schema_b = row.schema.replace(row.snippet, new_snippet_b).strip()

    if 'switched' in row.index:
        new_switched_a = row.switched.replace(row.snippet, new_snippet_a).strip()
        new_switched_b = row.switched.replace(row.snippet, new_snippet_b).strip()

    if 'schema' in row.index and 'switched' in row.index:
        if row.correct_answer.lower() == 'a':
            return new_schema_a, new_schema_b, new_switched_b, new_switched_a
        return new_schema_b, new_schema_a, new_switched_a, new_switched_b
    elif 'schema' in row.index:
        if row.correct_answer.lower() == 'a':
            return new_schema_a, new_schema_b
        return new_schema_b, new_schema_a
    else:
        if row.correct_answer.lower() == 'a':
            return new_switched_b, new_switched_a
        return new_switched_a, new_switched_b


def generate_is_associative_column(df):
    with open(WINOGRAD_ASSOCIATIVE_LABEL_FILE, 'r') as fp:
        english_associative_json = json.load(fp)

    df['is_associative'] = False
    for item in english_associative_json:
        if item['is_associative'] == 1:
            df.loc[item['index'], 'is_associative'] = True

    return df


def generate_is_switchable_column(df):
    with open(WINOGRAD_SWITCHED_LABEL_FILE, 'r') as fp:
        english_switched_json = json.load(fp)

    df['is_switchable'] = False
    for item in english_switched_json:
        if item['is_switchable'] == 1:
            df.loc[item['index'], 'is_switchable'] = True

    # these questions were not present in english switchable label file
    extra_switchable_indexes = [277, 278, 279, 280]
    for index in extra_switchable_indexes:
        df.loc[index, 'is_switchable'] = True

    return df


def minimize_substitution_range(subs_a, subs_b):
    i = 1
    while subs_b[:i] == subs_a[:i]:
        i += 1
    if subs_b[i - 1] == ' ':
        subs_a = subs_a[i - 1:]
        subs_b = subs_b[i - 1:]

    return subs_a, subs_b


def apply_substitution_exceptions(subs_a, subs_b):
    if subs_b == 'o cara que vestia uma farda':
        subs_a = 'o jim'
        subs_b = 'um cara que vestia uma farda e tinha uma grande barba ruiva'
    if subs_b == 'o homem' and subs_a == 'john':
        subs_b = 'um homem'
    if subs_a == 'o desenho do sam':
        subs_a = 'do sam'
        subs_b = 'da tina'
    if subs_a == 'o homem' and subs_b == 'o filho':
        subs_a = 'homem'
        subs_b = 'filho'
    if subs_a == 'Goodman':
        subs_a = 'Sam Goodman'

    return subs_a, subs_b


def capitalize_each_sentence(text):
    p = re.compile(r'((?<=[\.\?!]\s)(\w+)|(^\w+))')
    text = p.sub(cap, text)

    return text


def manually_generate_switched_sentence(row):
    if not row.is_switchable:
        return ''

    switched = row.schema.lower()
    subs_a, subs_b = minimize_substitution_range(row.substitution_a.lower(),
                                                 row.substitution_b.lower())
    subs_a, subs_b = apply_substitution_exceptions(subs_a, subs_b)

    switched = switched.replace(subs_a, '<PLACEHOLDER>')\
                       .replace(subs_b, subs_a)\
                       .replace('<PLACEHOLDER>', subs_b)\
                       .replace('seu homem', 'o homem')

    switched = capitalize_each_sentence(switched)

    return switched


def fill_df_from_english_switched_json(df):
    with open(WINOGRAD_SWITCHED_LABEL_FILE, 'r') as fp:
        english_switched_json = json.load(fp)

    for item in english_switched_json:
        df.loc[item['index'], 'substitution_a'] = item['answer0']
        df.loc[item['index'], 'substitution_b'] = item['answer1']
        df.loc[item['index'], 'correct_answer'] = 'A' if item['answer0'] == item['correct_answer'] else 'B'
        pronoun = re.findall(r'\[(.*?)\]', item['sentence_switched'])
        df.loc[item['index'], 'pronoun'] = pronoun
        df.loc[item['index'], 'snippet'] = '[' + pronoun[0] + ']'
        df.loc[item['index'], 'switched'] = item['sentence_switched']

    return df


def cap(match):
    return(match.group().capitalize())


def load_capitalized_words():
    with open(CAPITALIZED_WORD_LIST_FILE, 'r') as capitalized_words_file:
        capitalized_words = [line.strip() for line in capitalized_words_file.readlines()]

    return capitalized_words


def capitalize_words(sentence, capitalized_words, english):
    words = []

    for word in custom_tokenizer(sentence, english):
        if word.capitalize() in capitalized_words or (len(words) >= 1 and words[-1][-1] in ['.', '!', '?']):
            word = word.capitalize()
        if len(words) >= 1 and words[-1] == '``':
            words[-1] = '"' + word
        elif word in ['.', ',', '!', '?', ';', "''"]:
            if word == "''":
                word = '"'
            words[-1] += word
        else:
            words += [word]
    sentence = ' '.join(words).strip()
    sentence = sentence.replace('" eu primeiro! "', '"Eu primeiro"!')
    sentence = sentence.replace('" Eu primeiro! "', '"Eu primeiro"!')
    sentence = sentence.replace('tv', 'TV')
    sentence = sentence.replace('tv.', 'TV.')

    return sentence


def add_manual_fixes(df):
    with open(MANUAL_PT_FIXES_FILE, 'r', encoding='utf8') as fp:
        manual_fixes_json = json.load(fp)

    for item in manual_fixes_json:
        df.loc[item['question_id'], 'manually_fixed_correct_sentence'] = item['manually_fixed_correct_sentence']
        df.loc[item['question_id'], 'manually_fixed_incorrect_sentence'] = item['manually_fixed_incorrect_sentence']
        if 'manually_fixed_correct_switched' in item:
            df.loc[item['question_id'], 'manually_fixed_correct_switched'] = item['manually_fixed_correct_switched']
            df.loc[item['question_id'], 'manually_fixed_incorrect_switched'] = item['manually_fixed_incorrect_switched']

    return df


def prepare_full_json():
    english = False if PORTUGUESE else True

    if not english:
        df = generate_df()
    else:
        df = generate_df_from_json()

    df = generate_is_switchable_column(generate_is_associative_column(df))

    if english:
        df = fill_df_from_english_switched_json(df)
        df['correct_switched'], df['incorrect_switched'] = zip(*df.apply(generate_full_sentences, axis=1))
        df['translated'] = True
    else:
        df['switched'] = df.apply(manually_generate_switched_sentence, axis=1)
        df['correct_sentence'], df['incorrect_sentence'], df['correct_switched'], df['incorrect_switched'] = \
            zip(*df.apply(generate_full_sentences, axis=1))

    df['correct_sentence'] = df['correct_sentence'].apply(capitalize_words)
    df['incorrect_sentence'] = df['incorrect_sentence'].apply(capitalize_words)
    df['correct_switched'] = df['correct_switched'].apply(capitalize_words)
    df['incorrect_switched'] = df['incorrect_switched'].apply(capitalize_words)

    df['manually_fixed_correct_sentence'], df['manually_fixed_incorrect_sentence'], \
        df['manually_fixed_correct_switched'], df['manually_fixed_incorrect_switched'] = \
        df['correct_sentence'], df['incorrect_sentence'], df['correct_switched'], df['incorrect_switched']

    if not english:
        df = add_manual_fixes(df)

    generate_json(df)