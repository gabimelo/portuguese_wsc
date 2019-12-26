import numpy as np

from src.language_model_usage.generation import generate
from src.helpers.logger import Logger
from src.winograd_collection_manipulation.text_manipulation import get_vocab_list, custom_tokenizer

logger = Logger()


def analyse_single_wsc(model_file_name, corpus, device, correct_sentence, wrong_sentence, partial=False):
    _, correct_words_probs = generate(model_file_name, corpus, device, input_wsc=correct_sentence)
    _, wrong_words_probs = generate(model_file_name, corpus, device, input_wsc=wrong_sentence)

    if partial:
        for i in range(len(correct_sentence.split())):
            if correct_sentence.split()[-i-1] != wrong_sentence.split()[-i-1]:  # noqaE226
                break
        correct_words_probs = correct_words_probs[-i:]
        wrong_words_probs = wrong_words_probs[-i:]

    if np.prod(correct_words_probs) > np.prod(wrong_words_probs):
        return True
    elif np.prod(correct_words_probs) == np.prod(wrong_words_probs):
        return False
    else:
        return False


def find_missing_wsc_words_in_corpus_vocab(df, corpus, english=False):
    wsc_vocab = set()
    for col in df.loc[:, (df.applymap(type) == str).all(0)].columns:
        vocab = get_vocab_list(df[col].values, english, for_model=True)
        wsc_vocab.update(vocab)

    missing_words = []
    for word in wsc_vocab:
        if word not in corpus.dictionary.word2idx:
            missing_words.append(word)

    return missing_words


def winograd_test(df, corpus, model_file_name, device, english=False):
    if 'translated' in df:
        df.drop(df[~df.translated].index, inplace=True)

    def sentence_to_word_list(sentence):
        word_list = custom_tokenizer(sentence, english, for_model=True)
        unknown_word = '<unk>' if '<unk>' in corpus.dictionary.word2idx else '<UNK>'
        word_list = [word if word not in missing_words else unknown_word for word in word_list]

        return word_list

    def test_row(correct_sentence, incorrect_sentence, partial):
        winograd_sentences = ((' ').join(sentence_to_word_list(correct_sentence)).strip(),
                              (' ').join(sentence_to_word_list(incorrect_sentence)).strip())
        return analyse_single_wsc(model_file_name, corpus, device,
                                  winograd_sentences[0], winograd_sentences[1], partial)

    def test_set(df, correct_sentence_column, incorrect_sentence_column, result_column, partial, subset_key):
        if subset_key is not None and subset_key[0] != '~':
            df.loc[df[subset_key], result_column] = df[df[subset_key]].apply(lambda x:
                                                                             test_row(x[correct_sentence_column],
                                                                                      x[incorrect_sentence_column],
                                                                                      partial),
                                                                             axis=1)
            accuracy = (df[df[subset_key]][result_column]).sum() / len(df[df[subset_key]])
            logger.info('Accuracy: {} for test run on {} examples'.format(accuracy, len(df[df[subset_key]])))
        elif subset_key is not None:
            subset_key = subset_key[1:]
            df.loc[~df[subset_key], result_column] = df[~df[subset_key]].apply(lambda x:
                                                                               test_row(x[correct_sentence_column],
                                                                                        x[incorrect_sentence_column],
                                                                                        partial),
                                                                               axis=1)
            accuracy = (df[~df[subset_key]][result_column]).sum() / len(df[~df[subset_key]])
            logger.info('Accuracy: {} for test run on {} examples'.format(accuracy, len(df[~df[subset_key]])))
        else:
            df[result_column] = df.apply(lambda x: test_row(x[correct_sentence_column],
                                                            x[incorrect_sentence_column],
                                                            partial), axis=1)
            accuracy = (df[result_column]).sum() / len(df)
            logger.info('Accuracy: {} for test run on {} examples'.format(accuracy, len(df)))

    def test_full_and_partial(test_name, df, correct_sentence_column,
                              incorrect_sentence_column, result_column, subset_key=None):
        logger.info(test_name + ', full scoring')
        test_set(df, correct_sentence_column, incorrect_sentence_column,
                 result_column + '_full', False, subset_key)
        logger.info(test_name + ', partial scoring')
        test_set(df, correct_sentence_column, incorrect_sentence_column,
                 result_column + '_partial', True, subset_key)

    def run_full_set_of_tests():
        result_column = 'test_result'
        df[result_column + '_full'] = df[result_column + '_partial'] = False
        test_full_and_partial('Test on full set', df, 'correct_sentence',
                              'incorrect_sentence', result_column)

        if 'manually_fixed_correct_sentence' in df.columns and df.iloc[0]['manually_fixed_correct_sentence'] != '':
            result_column = 'test_result_manually_fixed'
            df[result_column + '_full'] = df[result_column + '_partial'] = False
            test_full_and_partial('Test on full set, with substitutions manually fixed',
                                  df, 'manually_fixed_correct_sentence', 'manually_fixed_incorrect_sentence',
                                  result_column)

        if 'is_switchable' in df.columns:
            result_column = 'test_result_switchable_switched'
            df[result_column + '_full'] = df[result_column + '_partial'] = False
            test_full_and_partial('Test on switchable set, switched',
                                  df, 'correct_switched', 'incorrect_switched',
                                  result_column, 'is_switchable')

            result_column = 'test_result_switchable_unswitched'
            df[result_column + '_full'] = df[result_column + '_partial'] = False
            test_full_and_partial('Test on switchable set, unswitched',
                                  df, 'correct_sentence', 'incorrect_sentence',
                                  result_column, 'is_switchable')

            consistency_full = (df[df.is_switchable]['test_result_switchable_switched_full'] ==
                                df[df.is_switchable]['test_result_switchable_unswitched_full']) \
                               .sum() / len(df.is_switchable)  # noqa E127
            logger.info('Consistency for full scoring: {}'.format(consistency_full))
            consistency_partial = (df[df.is_switchable]['test_result_switchable_switched_partial'] ==
                                   df[df.is_switchable]['test_result_switchable_unswitched_partial']) \
                                  .sum() / len(df.is_switchable)  # noqa E127
            logger.info('Consistency for partial scoring: {}'.format(consistency_partial))

        if 'is_associative' in df.columns:
            result_column = 'test_result_associative'
            df[result_column + '_full'] = df[result_column + '_partial'] = False
            test_full_and_partial('Test on associative set',
                                  df, 'correct_sentence', 'incorrect_sentence',
                                  result_column, 'is_associative')

            result_column = 'test_result_non_associative'
            df[result_column + '_full'] = df[result_column + '_partial'] = False
            test_full_and_partial('Test on non-associative set',
                                  df, 'correct_sentence', 'incorrect_sentence',
                                  result_column, '~is_associative')

            if 'manually_fixed_correct_sentence' in df.columns and \
               df.iloc[0]['manually_fixed_correct_sentence'] != '':
                result_column = 'manually_fixed_test_result_associative'
                df[result_column + '_full'] = df[result_column + '_partial'] = False
                test_full_and_partial('Test on associative set, with substitutions manually fixed',
                                      df, 'manually_fixed_correct_sentence',
                                      'manually_fixed_incorrect_sentence',
                                      result_column, 'is_associative')

                result_column = 'manually_fixed_test_result_non_associative'
                df[result_column + '_full'] = df[result_column + '_partial'] = False
                test_full_and_partial('Test on non-associative set, with substitutions manually fixed',
                                      df, 'manually_fixed_correct_sentence',
                                      'manually_fixed_incorrect_sentence',
                                      result_column, '~is_associative')

    missing_words = find_missing_wsc_words_in_corpus_vocab(df, corpus, english)
    run_full_set_of_tests()

    return df
