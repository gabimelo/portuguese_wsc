# coding: utf-8

import os

import torch
import torch.nn as nn
import pickle

from src.consts import (
    RANDOM_SEED, USE_CUDA, MODEL_TYPE, EMBEDDINGS_SIZE, HIDDEN_UNIT_COUNT,
    LAYER_COUNT, DROPOUT_PROB, TIED, CORPUS_FILE_NAME
)
from src.corpus import Corpus
from src.model import RNNModel
from src.logger import Logger
from src.custom_data_parallel import CustomDataParallel
from src.training import train

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


def main():
    setup_torch()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    corpus = get_corpus()

    ntokens = len(corpus.dictionary)
    model = RNNModel(MODEL_TYPE, ntokens, EMBEDDINGS_SIZE, HIDDEN_UNIT_COUNT, LAYER_COUNT, DROPOUT_PROB,
                     TIED)
    model = CustomDataParallel(model)
    criterion = nn.CrossEntropyLoss()

    train(model, corpus, criterion, device)


if __name__ == '__main__':
    main()
