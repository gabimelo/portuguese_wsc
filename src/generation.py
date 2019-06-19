import torch
import torch.nn.functional as F

from src.logger import Logger
from src.consts import WORDS_TO_GENERATE, TEMPERATURE
from src.utils import permute_for_parallelization, get_results_from_data_parallelized_forward

logger = Logger()


def generate(model_file_name, corpus, ntokens, device, input_wsc=None):
    with open(model_file_name, 'rb') as f:
        model = torch.load(f).to(device)
    use_data_paralellization = True if type(model).__name__ == 'CustomDataParallel' else False

    model.eval()

    batch_size = 1
    hidden = model.init_hidden(batch_size)
    # TODO test this new generation of first word
    # if hasattr(corpus.dictionary, 'word_count'):
    #     input_word_id = torch.multinomial(corpus.dictionary.word_count, 1)[0]
    # else:
    input_word_id = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

    if input_wsc is not None:
        input_wsc_words = input_wsc.split()
        input_word_id.fill_(corpus.dictionary.word2idx[input_wsc_words[0]])

    input_words = [corpus.dictionary.idx2word[input_word_id]]
    # TODO this initial prob should be obtained from distribution (frequency
    #  count of word) instead of being 1
    input_words_probs = [1]

    number_of_words = WORDS_TO_GENERATE if input_wsc is None else len(input_wsc_words) - 1

    with torch.no_grad():  # no tracking history
        for i in range(number_of_words):
            if use_data_paralellization:
                hidden, input_word_id = permute_for_parallelization(hidden, input_word_id)
                results = model(input_word_id, hidden)
                outputs, hidden = get_results_from_data_parallelized_forward(results, device)
                hidden = permute_for_parallelization(hidden)
                output = outputs[0]
            else:
                output, hidden = model(input_word_id, hidden)

#             word_weights = output.squeeze().div(TEMPERATURE).exp().cpu()
            word_probs = F.softmax(output.squeeze().div(TEMPERATURE), dim=0)

            if input_wsc is None:
                new_word_id = torch.multinomial(word_probs, 1)[0]
            else:
                new_word_id = corpus.dictionary.word2idx[input_wsc_words[i + 1]]

            input_word_id.fill_(new_word_id)
            input_words.append(corpus.dictionary.idx2word[new_word_id])
            input_words_probs.append(word_probs[new_word_id])

    return input_words, input_words_probs
