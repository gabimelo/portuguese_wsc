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


def add_at_marks(word_list):
    for i, word in enumerate(word_list):
        if '@' not in word and len(word) > 1:
            for punct in string.punctuation:
                if punct != "'" and punct != '@':
                    word_list[i] = word_list[i].replace(punct, '@' + punct + '@')

    for i in range(len(word_list) - 1, -1, -1):
        if '@' in word_list[i]:
            split_on = word_list[i][word_list[i].index('@'):word_list[i].rindex('@') + 1]
            word_list[i:i + 1] = [word_list[i].split(split_on)[0], split_on, word_list[i].split(split_on)[1]]

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
