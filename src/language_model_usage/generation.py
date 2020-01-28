import torch
import torch.nn.functional as F

from src.helpers.logger import Logger
from src.helpers.consts import WORDS_TO_GENERATE, TEMPERATURE
from src.helpers.utils import load_model
from src.modeling.utils import permute_for_parallelization, get_results_from_data_parallelized_forward

logger = Logger()


def generate(model_file_name, corpus, device, input_wsc=None, model=None):
    model = model or load_model(model_file_name, device)

    use_data_paralellization = True if type(model).__name__ == 'CustomDataParallel' else False

    model.eval()

    batch_size = 1
    hidden = model.init_hidden(batch_size)

    word_frequency = torch.tensor(list(corpus.dictionary.word_count.values()), dtype=torch.float)

    if input_wsc is not None:
        input_wsc_words = input_wsc.split()
        input_word_id = (
            torch.tensor([[
                corpus.dictionary.word2idx[input_wsc_words[0]]
            ]]).to(device)
        )
    else:
        input_word_id = (
            torch.tensor([[
                torch.multinomial(word_frequency, 1)[0]
            ]]).to(device)
        )

    input_words_probs = [
        (
            corpus.dictionary.word_count[
                corpus.dictionary.idx2word[input_word_id]
            ] /
            word_frequency.sum()
        ).
        item()
    ]

    input_words = [corpus.dictionary.idx2word[input_word_id]]

    number_of_words = (WORDS_TO_GENERATE if input_wsc is None else len(input_wsc_words)) - 1

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

            logits = model.decoder(output)
            word_probs = F.softmax(logits.squeeze().div(TEMPERATURE), dim=0)

            if input_wsc is None:
                new_word_id = torch.multinomial(word_probs, 1)[0]
            else:
                new_word_id = corpus.dictionary.word2idx[input_wsc_words[i + 1]]

            input_word_id.fill_(new_word_id)
            input_words.append(corpus.dictionary.idx2word[new_word_id])
            input_words_probs.append(word_probs[new_word_id].item())

    return input_words, input_words_probs
