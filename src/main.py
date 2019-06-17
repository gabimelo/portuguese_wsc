# coding: utf-8
import os

import torch
import torch.nn as nn
import pickle

from src.consts import (
    RANDOM_SEED, USE_CUDA, MODEL_TYPE, EMBEDDINGS_SIZE, HIDDEN_UNIT_COUNT,
    LAYER_COUNT, DROPOUT_PROB, TIED, CORPUS_FILE_NAME, USE_DATA_PARALLELIZATION
)
from src.corpus import Corpus
from src.model import RNNModel
from src.logger import Logger
from src.custom_data_parallel import CustomDataParallel
from src.training import train
from src.generation import generate
from src.utils import get_latest_model_file, summary
from src.parallel import DataParallelCriterion

logger = Logger()


def setup_torch():
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        if not USE_CUDA:
            logger.info("WARNING: You have a CUDA device, so you should probably run with --cuda")


def get_corpus():
    if os.path.exists(CORPUS_FILE_NAME):
        corpus = pickle.load(open(CORPUS_FILE_NAME, "rb"))
    else:
        corpus = Corpus()
        corpus.add_corpus_data()

    return corpus


def main(training=True, use_data_paralellization=False, model_timestamp=None, verbose=False):
    setup_torch()
    # code seems to run slower (~90ms/batch, with batch_size=40) when default GPU is not cuda:0
    main_gpu_index = 0  # TODO set this somewhere else
    device = torch.device("cuda:" + str(main_gpu_index) if USE_CUDA else "cpu")
    corpus = get_corpus()
    ntokens = len(corpus.dictionary)

    # TODO remove these two lines
    assert ntokens == 111550
    assert corpus.valid.size()[0] == 11606861
    assert corpus.train.max() < ntokens
    assert corpus.valid.max() < ntokens
    assert corpus.test.max() < ntokens

    if training:
        model = RNNModel(MODEL_TYPE, ntokens, EMBEDDINGS_SIZE, HIDDEN_UNIT_COUNT, LAYER_COUNT, DROPOUT_PROB,
                         TIED).to(device)
        criterion = nn.CrossEntropyLoss()

        if use_data_paralellization or USE_DATA_PARALLELIZATION:
            cuda_devices = [i for i in range(torch.cuda.device_count())]
            device_ids = [main_gpu_index] + cuda_devices[:main_gpu_index] + cuda_devices[main_gpu_index + 1:]
            model = CustomDataParallel(model, device_ids=device_ids)
            criterion = DataParallelCriterion(criterion, device_ids=device_ids)

        # TODO use global Learning Rate here
        # TODO check if weight decay will be kept
#         optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay=1.2e-6)
#         optimizer = torch.optim.SGD(model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay=1.2e-6)
        optimizer = None

        if verbose:
            summary(model, criterion)

        train(model, corpus, criterion, optimizer, device, use_data_paralellization or USE_DATA_PARALLELIZATION)
    else:
        if model_timestamp is None:
            model_file_name = get_latest_model_file()

        logger.info('Generating text')
        words, words_probs = generate(model_file_name, corpus, ntokens, device, is_wsc=False)
        logger.info('Generated text: ', (' ').join(words))


if __name__ == '__main__':
    main()
