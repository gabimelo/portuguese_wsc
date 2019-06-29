import string

from nltk.tokenize import word_tokenize


def clean_quotes(word_list):
    while "''" in word_list:
        i = word_list.index("''")
        word_list[i] = '"'
    while "``" in word_list:
        i = word_list.index("``")
        word_list[i] = '"'

    return word_list


def clean_at_marks(word_list):
    "For merging things like ['@', '-', '@'] into ['@-@']"
    start = 0
    while '@' in word_list[start:]:
        i = word_list.index('@')
        if word_list[i + 2] == '@':
            middle = word_list.pop(i + 1)
            word_list.pop(i + 1)
            word_list[i] = '@' + middle + '@'
        start = i

    return word_list


def add_at_marks(word_list):
    for i, word in enumerate(word_list):
        if '@' not in word and len(word) > 1:
            for punct in string.punctuation:
                if punct != "'":
                    word_list[i] = word.replace(punct, '@' + punct + '@')
    return word_list


def custom_tokenizer(sentence, english, for_model=False):
    if not english:
        word_list = word_tokenize(sentence, language='portuguese')
    else:
        sentence = sentence.replace("n't", "n 't")
        word_list = word_tokenize(sentence, language='english')
        word_list = clean_quotes(word_list)
        if for_model:
            word_list = add_at_marks(word_list)

    return word_list


def get_vocab_list(sentence_list, english):
    return [word for sentence in sentence_list
            for word in custom_tokenizer(sentence, english)]
