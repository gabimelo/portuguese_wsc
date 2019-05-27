import numpy as np

from src.generation import generate


def analyse_single_wsc(model_file_name, corpus, ntokens, device, correct_sentence, wrong_sentence):
    _, correct_words_probs = generate(model_file_name, corpus, ntokens, device, input_wsc=correct_sentence)
    _, wrong_words_probs = generate(model_file_name, corpus, ntokens, device, input_wsc=wrong_sentence)
    if np.prod(correct_words_probs) >= np.prod(wrong_words_probs):
        return True
    else:
        return False
