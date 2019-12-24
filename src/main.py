# coding: utf-8
import os
import pickle
import click

import torch
import torch.nn as nn

from src.helpers.consts import (
    RANDOM_SEED, USE_CUDA, MODEL_TYPE, EMBEDDINGS_SIZE, HIDDEN_UNIT_COUNT,
    LAYER_COUNT, DROPOUT_PROB, TIED, CORPUS_FILE_NAME, USE_DATA_PARALLELIZATION
)
from src.datasets_manipulation.corpus import Corpus
from src.modeling.model import RNNModel
from src.helpers.logger import Logger
from src.modeling.custom_data_parallel import CustomDataParallel
from src.modeling.training import train
from src.language_model_usage.generation import generate
from src.helpers.utils import get_latest_model_file, summary, log_loaded_model_info
from src.modeling.parallel import DataParallelCriterion
from src.winograd_collection_manipulation.wsc_json_handler import generate_df_from_json
from src.language_model_usage.winograd_schema_challenge import winograd_test
from src.helpers.consts import PORTUGUESE, MAIN_GPU_INDEX

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


def sanity_checks(corpus, ntokens):
    assert corpus.train.max() < ntokens
    assert corpus.valid.max() < ntokens
    assert corpus.test.max() < ntokens


@click.command()
@click.option('--training', is_flag=True)
@click.option('--generating', is_flag=True)
@click.option('--model_file_name', default=None)
@click.option('--quiet', is_flag=True)
def main(training, generating, model_file_name, quiet):
    verbose = not quiet

    setup_torch()
    # code seems to run slower (~90ms/batch, with batch_size=40) when default GPU is not cuda:0
    device = torch.device("cuda:" + str(MAIN_GPU_INDEX) if USE_CUDA else "cpu")
    corpus = get_corpus()
    ntokens = len(corpus.dictionary)

    sanity_checks(corpus, ntokens)

    if training:
        model = (
            RNNModel(
                MODEL_TYPE,
                ntokens,
                EMBEDDINGS_SIZE,
                HIDDEN_UNIT_COUNT,
                LAYER_COUNT,
                DROPOUT_PROB,
                TIED
            ).to(device)
        )
        criterion = nn.CrossEntropyLoss()

        if USE_DATA_PARALLELIZATION:
            cuda_devices = [i for i in range(torch.cuda.device_count())]
            device_ids = [MAIN_GPU_INDEX] + cuda_devices[:MAIN_GPU_INDEX] + cuda_devices[MAIN_GPU_INDEX + 1:]
            model = CustomDataParallel(model, device_ids=device_ids)
            criterion = DataParallelCriterion(criterion, device_ids=device_ids)

        # TODO use global Learning Rate here
        # TODO check if weight decay will be kept
#         optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay=1.2e-6)
#         optimizer = torch.optim.SGD(model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay=1.2e-6)
        optimizer = None

        if verbose:
            summary(model, criterion)

        train(model, corpus, criterion, optimizer, device, USE_DATA_PARALLELIZATION)
    else:
        if model_file_name is None:
            model_file_name = get_latest_model_file()

        if verbose:
            log_loaded_model_info(model_file_name, device)

        if not generating:
            logger.info('Generating WSC set, using model: {}'.format(model_file_name))
            df = generate_df_from_json()
            df = winograd_test(df, corpus, model_file_name, ntokens, device, english=not PORTUGUESE)
        else:
            logger.info('Generating text, using model: {}'.format(model_file_name))
            words, words_probs = generate(model_file_name, corpus, ntokens, device)
            logger.info('Generated text: {}'.format((' ').join(words)))


if __name__ == '__main__':
    main()
