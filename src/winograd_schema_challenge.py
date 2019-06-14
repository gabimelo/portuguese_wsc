import re

import numpy as np
from nltk.tokenize import word_tokenize

from src.generation import generate


def analyse_single_wsc(model_file_name, corpus, ntokens, device, correct_sentence, wrong_sentence, partial=False):
    _, correct_words_probs = generate(model_file_name, corpus, ntokens, device, input_wsc=correct_sentence)
    _, wrong_words_probs = generate(model_file_name, corpus, ntokens, device, input_wsc=wrong_sentence)
    
    if partial:
        for i in range(len(correct_sentence.split())):
            if correct_sentence.split()[-i-1] != wrong_sentence.split()[-i-1]:
                break
        correct_words_probs = correct_words_probs[-i:]
        wrong_words_probs = wrong_words_probs[-i:]
    
    if np.prod(correct_words_probs) >= np.prod(wrong_words_probs):
        return True
    else:
        return False


def find_missing_wsc_words_in_corpus_vocab(df, corpus):
    correct_sentences_vocab = [ word for sentence in df.correct_sentence.values 
                               for word in word_tokenize(sentence, language='portuguese')]
    incorrect_sentences_vocab = [ word for sentence in df.incorrect_sentence.values 
                                 for word in word_tokenize(sentence, language='portuguese')]
    wsc_vocab = set(correct_sentences_vocab + incorrect_sentences_vocab)
    # TODO quotes need to be in the same format as wiki corpus, and split from words
    missing_words = []
    for word in wsc_vocab:
        if word not in corpus.dictionary.word2idx:
            missing_words.append(word)
            
    return missing_words


def generate_full_sentences(row):
    if row.pronoun.islower() and (row.substitution_b.lower() in row.schema or 
                                  row.substitution_b.lower() not in row.schema.lower()):
        subs_a = row.substitution_a.lower()
        subs_b = row.substitution_b.lower()
    else:
        subs_a = row.substitution_a
        subs_b = row.substitution_b

    new_snippet_a = re.sub(r"\b%s\b" % row.pronoun , subs_a, row.snippet)
    new_snippet_b = re.sub(r"\b%s\b" % row.pronoun , subs_b, row.snippet)
    new_schema_a = row.schema.replace(row.snippet, new_snippet_a).strip()
    new_schema_b = row.schema.replace(row.snippet, new_snippet_b).strip()
    
    if row.correct_answer.lower() == 'a':
        return new_schema_a, new_schema_b
    return new_schema_b, new_schema_a


def winograd_test(df, corpus, model_file_name, ntokens, device, partial=False):
    def sentence_to_word_list(sentence):
        word_list = word_tokenize(sentence, language='portuguese')
        word_list = [ word if word not in missing_words else '<UNK>' for word in word_list]

        return word_list

    def run_test(row):
        winograd_sentences = ((' ').join(sentence_to_word_list(row.correct_sentence)).strip(),
                              (' ').join(sentence_to_word_list(row.incorrect_sentence)).strip())
        return analyse_single_wsc(model_file_name, corpus, ntokens, device, winograd_sentences[0], winograd_sentences[1], partial)

    missing_words = find_missing_wsc_words_in_corpus_vocab(df, corpus)
    df['test_result'] = df.apply(run_test, axis=1)
    
    accuracy = sum(df.test_result) / len(df)
    
    return df, accuracy
