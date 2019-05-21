# coding: utf-8
# import argparse
import math
# import os
import torch
import torch.nn as nn
# import torch.onnx

from src.consts import (
    RANDOM_SEED, USE_CUDA, PROCESSED_DATA_DIR_NAME, BATCH_SIZE, MODEL_TYPE, EMBEDDINGS_SIZE, HIDDEN_UNIT_COUNT,
    LAYER_COUNT, DROPOUT_PROB, TIED, SEQUENCE_LENGTH, EVAL_BATCH_SIZE, INITIAL_LEARNING_RATE, EPOCHS, GRADIENT_CLIPPING,
    LOG_INTERVAL, MODEL_FILE_NAME
)
from src.data_manipulation.text_manipulations import Corpus
from src.model import RNNModel
from src.logger import Logger
from src.training import train

logger = Logger()


def setup_torch():
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        if not USE_CUDA:
            logger.info("WARNING: You have a CUDA device, so you should probably run with --cuda")


def main():
    setup_torch()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    corpus = Corpus()
    corpus.add_corpus_data()

    ntokens = len(corpus.dictionary)
    model = RNNModel(MODEL_TYPE, ntokens, EMBEDDINGS_SIZE, HIDDEN_UNIT_COUNT, LAYER_COUNT, DROPOUT_PROB,
                     TIED).to(device)
    criterion = nn.CrossEntropyLoss()

    train(model, corpus, criterion, device)


if __name__ == '__main__':
    main()
