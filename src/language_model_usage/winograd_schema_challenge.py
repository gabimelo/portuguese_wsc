from functools import partial

import numpy as np
import torch

from src.language_model_usage.generation import generate
from src.helpers.logger import Logger
from src.winograd_collection_manipulation.text_manipulation import custom_tokenizer

logger = Logger()


def get_probability_of_next_sentence(tokenizer, model, text1, text2):
    text1_tokens = ['[CLS]'] + tokenizer.tokenize(text1) + ['[SEP]']
    text2_tokens = tokenizer.tokenize(text2) + ['[SEP]']
    text = text1_tokens + text2_tokens
    indexed_tokens = tokenizer.convert_tokens_to_ids(text)
    segments_ids = [0] * len(text1_tokens) + [1] * len(text2_tokens)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    model.eval()
    prediction = model(tokens_tensor, token_type_ids=segments_tensors)
    prediction = prediction[0]  # tuple to tensor
    softmax = torch.nn.Softmax(dim=1)
    prediction_sm = softmax(prediction)

    return prediction_sm[0]


def get_sentence_breaks(first_sentence, second_sentence):
    for i in range(len(first_sentence.split())):
        if first_sentence.split()[i] != second_sentence.split()[i]:  # noqaE226
            break
    return i


def analyse_single_wsc_bert(model, tokenizer, correct_sentence, wrong_sentence):
    if correct_sentence == '' or wrong_sentence == '':
        return False, False

    i = get_sentence_breaks(correct_sentence, wrong_sentence)

    text1 = ' '.join(correct_sentence.split()[:i])
    text2 = ' '.join(correct_sentence.split()[i:])
    prob_correct_sentence_correct = get_probability_of_next_sentence(tokenizer, model, text1, text2)[0]

    text1 = ' '.join(wrong_sentence.split()[:i])
    text2 = ' '.join(wrong_sentence.split()[i:])
    prob_wrong_sentence_correct = get_probability_of_next_sentence(tokenizer, model, text1, text2)[0]

    result = prob_correct_sentence_correct.item() > prob_wrong_sentence_correct.item()

    return result, 0  # let's always return 0 for partial result


def get_partial_probs(correct_sentence, wrong_sentence, correct_words_probs, wrong_words_probs):
    for i in range(len(correct_sentence.split())):
        if correct_sentence.split()[-i-1] != wrong_sentence.split()[-i-1]:  # noqaE226
            break
    correct_words_probs = correct_words_probs[-i:]
    wrong_words_probs = wrong_words_probs[-i:]

    return correct_words_probs, wrong_words_probs


def analyse_single_wsc(model, model_file_name, corpus, device, correct_sentence, wrong_sentence):
    if correct_sentence == '' or wrong_sentence == '':
        return False, False

    _, correct_words_probs = generate(model_file_name, corpus, device, input_wsc=correct_sentence, model=model)
    _, wrong_words_probs = generate(model_file_name, corpus, device, input_wsc=wrong_sentence, model=model)

    correct_words_probs_partial, wrong_words_probs_partial = \
        get_partial_probs(correct_sentence, wrong_sentence, correct_words_probs, wrong_words_probs)

    full_result = np.prod(correct_words_probs) > np.prod(wrong_words_probs)
    partial_result = np.prod(correct_words_probs_partial) > np.prod(wrong_words_probs_partial)

    return full_result, partial_result


def find_missing_wsc_words_in_corpus_vocab(df, text_columns, corpus, english=False):
    wsc_vocab = set(df[text_columns].applymap(lambda x: custom_tokenizer(x, english, for_model=True)).sum().sum())
    missing_words = list(wsc_vocab - set(corpus.dictionary.word2idx))

    return missing_words


def prepare_text(unknown_word_token, missing_words, english, input_text):
    word_list = custom_tokenizer(input_text, english, for_model=True)
    word_list = [word if word not in missing_words else unknown_word_token for word in word_list]
    updated_text = (' ').join(word_list).strip()

    return updated_text


def prepare_text_cols(df, corpus, english):
    unknown_word_token = '<unk>' if '<unk>' in corpus.dictionary.word2idx else '<UNK>'
    text_columns = df.loc[:, (df.applymap(type) == str).all(axis=0)].columns
    missing_words = find_missing_wsc_words_in_corpus_vocab(df, text_columns, corpus, english)

    partial_prepare_text = partial(prepare_text, unknown_word_token, missing_words, english)

    df[text_columns] = df[text_columns].applymap(partial_prepare_text)

    return df


def run_test_for_col(df, partial_func, result_col):
    if result_col == 'original':
        correct_column = 'correct_sentence'
        incorrect_column = 'incorrect_sentence'
    elif result_col == 'switched':
        correct_column = 'correct_switched'
        incorrect_column = 'incorrect_switched'
    else:
        correct_column = 'manually_fixed_correct_sentence'
        incorrect_column = 'manually_fixed_incorrect_sentence'

    for i, row in df.iterrows():
        df.loc[i, f'{result_col}_result_full'], df.loc[i, f'{result_col}_result_partial'] = \
            partial_func(row[correct_column], row[incorrect_column])

    return df


def calculate_single_accuracy(df_subset, result_column):
    accuracy = (df_subset[result_column]).sum() / len(df_subset)

    return accuracy


def calculate_accuracy(df, subset_key, result_column):
    if subset_key == '':
        df_subset = df.copy()
    elif subset_key[0] == '~':
        df_subset = df[~df[subset_key[1:]]].copy()
    else:
        df_subset = df[df[subset_key]].copy()

    return {
        'accuracy_full': calculate_single_accuracy(df_subset, f'{result_column}_result_full'),
        'accuracy_partial': calculate_single_accuracy(df_subset, f'{result_column}_result_partial'),
        'number_of_examples': len(df_subset),
    }


def calculate_consistency(df):
    df_switchable = df[df.is_switchable].copy()
    number_of_examples = len(df_switchable)

    number_of_consitents_full = \
        (df_switchable['original_result_full'] == df_switchable['switched_result_full']).sum()
    number_of_consitents_partial = \
        (df_switchable['original_result_partial'] == df_switchable['switched_result_partial']).sum()

    consistency_full = number_of_consitents_full / number_of_examples
    consistency_partial = number_of_consitents_partial / number_of_examples

    return {
        'consistency_full': consistency_full,
        'consistency_partial': consistency_partial,
        'number_of_examples': number_of_examples,
    }


def calculate_metrics(df, test_on_manually_fixed):
    metrics = {
        'complete': calculate_accuracy(df, '', 'original'),
        'associative': calculate_accuracy(df, 'is_associative', 'original'),
        'non_associative': calculate_accuracy(df, '~is_associative', 'original'),
        'switched': calculate_accuracy(df, 'is_switchable', 'switched'),
        'unswitched': calculate_accuracy(df, 'is_switchable', 'original'),
        'consistency': calculate_consistency(df),
    }

    if test_on_manually_fixed:
        metrics['manually_fixed_complete'] = calculate_accuracy(df, '', 'manually_fixed')
        metrics['manually_fixed_associative'] = calculate_accuracy(df, 'is_associative', 'manually_fixed')
        metrics['manually_fixed_non_associative'] = calculate_accuracy(df, '~is_associative', 'manually_fixed')

    return metrics


def generate_report(metrics):
    for metric_name, metric in metrics.items():
        if 'consistency' in metric_name:
            metric_type = 'Consistency'
            metric_key = 'consistency'
        else:
            metric_type = 'Accuracy'
            metric_key = 'accuracy'
        logger.info(
            f'{metric_type} for {metric_name} test on {metric["number_of_examples"]} examples: \n'
            f'full: {metric[f"{metric_key}_full"]} \n'
            f'partial: {metric[f"{metric_key}_partial"]} \n'
        )


def add_results_columns(df):
    df['original_result_full'] = False
    df['original_result_partial'] = False
    df['switched_result_full'] = False
    df['switched_result_partial'] = False
    df['manually_fixed_result_full'] = False
    df['manually_fixed_result_partial'] = False

    return df


def winograd_test(df, corpus, model_file_name, device, model, tokenizer, english=False, use_bert=False):
    df = df[df.translated].copy()
    df = prepare_text_cols(df, corpus, english)
    df = add_results_columns(df)

    if use_bert:
        partial_func = partial(analyse_single_wsc_bert, model, tokenizer)
    else:
        partial_func = partial(analyse_single_wsc, model, model_file_name, corpus, device)

    partial_run_test_for_col = partial(run_test_for_col, df, partial_func)

    df = partial_run_test_for_col(result_col='original')
    df = partial_run_test_for_col(result_col='switched')

    test_on_manually_fixed = (
        'manually_fixed_correct_sentence' in df.columns and
        df.iloc[0]['manually_fixed_correct_sentence'] != ''
    )
    if test_on_manually_fixed:
        df = partial_run_test_for_col(result_col='manually_fixed')

    metrics = calculate_metrics(df, test_on_manually_fixed)
    generate_report(metrics)
