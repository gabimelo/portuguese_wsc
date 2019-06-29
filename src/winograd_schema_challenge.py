import numpy as np

from src.generation import generate
from src.helpers.logger import Logger

logger = Logger()


def send_array_of_tensors_to_cpu(array_of_tensors):
    raise Exception('Check that this is not needed anymore!')


def analyse_single_wsc(model_file_name, corpus, ntokens, device, correct_sentence, wrong_sentence, partial=False):
    _, correct_words_probs = generate(model_file_name, corpus, ntokens, device, input_wsc=correct_sentence)
    _, wrong_words_probs = generate(model_file_name, corpus, ntokens, device, input_wsc=wrong_sentence)

    if partial:
        for i in range(len(correct_sentence.split())):
            if correct_sentence.split()[-i-1] != wrong_sentence.split()[-i-1]:  # noqaE226
                break
        correct_words_probs = correct_words_probs[-i:]
        wrong_words_probs = wrong_words_probs[-i:]

    correct_words_probs_cpu = send_array_of_tensors_to_cpu(correct_words_probs)
    wrong_words_probs_cpu = send_array_of_tensors_to_cpu(wrong_words_probs)

    if np.prod(correct_words_probs_cpu) > np.prod(wrong_words_probs_cpu):
        return True
    elif np.prod(correct_words_probs_cpu) == np.prod(wrong_words_probs_cpu):
        return False
    else:
        return False


def get_word_list():
    raise Exception('Should be importing from its new location')


def get_vocab_list():
    raise Exception('Should be importing from its new location')


def find_missing_wsc_words_in_corpus_vocab(df, corpus, english=False):
    # TODO check this is working
    wsc_vocab = set()
    for col in df.select_dtypes(include='str').columns:
        vocab = get_vocab_list(df.correct_sentence.values, english)
        wsc_vocab += vocab

    missing_words = []
    for word in wsc_vocab:
        if word not in corpus.dictionary.word2idx:
            missing_words.append(word)

    return missing_words


def winograd_test(df, corpus, model_file_name, ntokens, device, english=False):
    if 'translated' in df:
        df.drop(df[~df.translated].index, inplace=True)

    def sentence_to_word_list(sentence):
        word_list = get_word_list(sentence, english)
        unknown_word = '<unk>' if '<unk>' in corpus.dictionary.word2idx else '<UNK>'
        word_list = [word if word not in missing_words else unknown_word for word in word_list]

        return word_list

    def test_row(correct_sentence, incorrect_sentence, partial):
        winograd_sentences = ((' ').join(sentence_to_word_list(correct_sentence)).strip(),
                              (' ').join(sentence_to_word_list(incorrect_sentence)).strip())
        return analyse_single_wsc(model_file_name, corpus, ntokens, device,
                                  winograd_sentences[0], winograd_sentences[1], partial)

    def test_set(subset_of_df, correct_sentence_column, incorrect_sentence_column, result_column, partial):
        subset_of_df[result_column] = subset_of_df.apply(lambda x: test_row(x[correct_sentence_column],
                                                                            x[incorrect_sentence_column],
                                                                            partial), axis=1)
        accuracy = sum(subset_of_df[result_column]) / len(subset_of_df)
        logger.info('Accuracy: {} for test run on {} examples'.format(accuracy, len(subset_of_df)))

    def test_full_and_partial(test_name, subset_of_df, correct_sentence_column,
                              incorrect_sentence_column, result_column):
        logger.info(test_name + ', full scoring')
        test_set(subset_of_df, correct_sentence_column, incorrect_sentence_column,
                 result_column + '_full', False)
        logger.info(test_name + ', partial scoring')
        test_set(subset_of_df, correct_sentence_column, incorrect_sentence_column,
                 result_column + '_partial', True)

    def run_full_set_of_tests():
        result_column = 'test_result'
        df[result_column + '_full'] = df[result_column + '_partial'] = False
        test_full_and_partial('Test on full set', df, 'correct_sentence',
                              'incorrect_sentence', result_column)

        if 'manually_fixed_correct_sentence' in df.columns:
            result_column = 'test_result_manually_fixed'
            df[result_column + '_full'] = df[result_column + '_partial'] = False
            test_full_and_partial('Test on full set, with substitutions manually fixed',
                                  df, 'manually_fixed_correct_sentence', 'manually_fixed_incorrect_sentence',
                                  result_column)

        if 'is_switchable' in df.columns:
            result_column = 'test_result_switchable_switched'
            df[result_column + '_full'] = df[result_column + '_partial'] = False
            test_full_and_partial('Test on switchable set, switched',
                                  df[df.is_switchable], 'correct_switched', 'incorrect_switched',
                                  result_column)

            result_column = 'test_result_switchable_switched'
            df[result_column + '_full'] = df[result_column + '_partial'] = False
            test_full_and_partial('Test on switchable set, unswitched',
                                  df[df.is_switchable], 'correct_sentence', 'incorrect_sentence',
                                  result_column)

        if 'is_associative' in df.columns:
            result_column = 'test_result_associative'
            df[result_column + '_full'] = df[result_column + '_partial'] = False
            test_full_and_partial('Test on associative set',
                                  df[df.is_associative], 'correct_sentence', 'incorrect_sentence',
                                  result_column)

            result_column = 'test_result_non_associative'
            df[result_column + '_full'] = df[result_column + '_partial'] = False
            test_full_and_partial('Test on non-associative set',
                                  df[~df.is_associative], 'correct_sentence', 'incorrect_sentence',
                                  result_column)

            if 'manually_fixed_correct_sentence' in df.columns:
                result_column = 'manually_fixed_test_result_associative'
                df[result_column + '_full'] = df[result_column + '_partial'] = False
                test_full_and_partial('Test on associative set, with substitutions manually fixed',
                                      df[df.is_associative], 'manually_fixed_correct_sentence',
                                      'manually_fixed_incorrect_sentence',
                                      result_column)

                result_column = 'manually_fixed_test_result_non_associative'
                df[result_column + '_full'] = df[result_column + '_partial'] = False
                test_full_and_partial('Test on non-associative set, with substitutions manually fixed',
                                      df[~df.is_associative], 'manually_fixed_correct_sentence',
                                      'manually_fixed_incorrect_sentence',
                                      result_column)

    missing_words = find_missing_wsc_words_in_corpus_vocab(df, corpus, english)
    run_full_set_of_tests()

    return df
