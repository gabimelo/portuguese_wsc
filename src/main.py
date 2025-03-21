# coding: utf-8
import os
import pickle
import click

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForNextSentencePrediction

from src.helpers.consts import (
    RANDOM_SEED, USE_CUDA, MODEL_TYPE, EMBEDDINGS_SIZE, HIDDEN_UNIT_COUNT,
    LAYER_COUNT, DROPOUT_PROB, TIED, CORPUS_FILE_NAME, USE_DATA_PARALLELIZATION,
    PORTUGUESE, MAIN_GPU_INDEX
)
from src.datasets_manipulation.corpus import Corpus
from src.modeling.model import RNNModel
from src.helpers.logger import Logger
from src.modeling.custom_data_parallel import CustomDataParallel
from src.modeling.training import train
from src.language_model_usage.generation import generate
from src.helpers.utils import get_latest_model_file, summary, log_loaded_model_info, load_model
from src.modeling.parallel import DataParallelCriterion
from src.winograd_collection_manipulation.wsc_json_handler import generate_df_from_json
from src.language_model_usage.winograd_schema_challenge import winograd_test

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
@click.option('--use_bert', is_flag=True)
def main(training, generating, model_file_name, quiet, use_bert):
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

#         optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay=1.2e-6)
#         optimizer = torch.optim.SGD(model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay=1.2e-6)
        optimizer = None

        if verbose:
            summary(model, criterion)

        train(model, corpus, criterion, optimizer, device, USE_DATA_PARALLELIZATION)
    else:
        if not use_bert:
            if model_file_name is None:
                model_file_name = get_latest_model_file()
            model = load_model(model_file_name, device)
            if verbose:
                log_loaded_model_info(model_file_name, model, device)
            tokenizer = None
        else:
            # model_file_name = 'bert-base-multilingual-cased' if PORTUGUESE else 'bert-base-cased'
            # model_file_name = 'bert-base-multilingual-cased' if PORTUGUESE else 'bert-large-cased'
            # model_file_name = 'models/neuralmind/bert-base-portuguese-cased' if PORTUGUESE else 'bert-large-cased'
            model_file_name = 'models/neuralmind/bert-large-portuguese-cased' if PORTUGUESE else 'bert-large-cased'
            tokenizer = BertTokenizer.from_pretrained(model_file_name)
            model = BertForNextSentencePrediction.from_pretrained(model_file_name)

        if not generating:
            logger.info('Generating WSC set, using model: {}'.format(model_file_name))
            df = generate_df_from_json()
            df = winograd_test(
                df, corpus, model_file_name, device, model, tokenizer,
                english=not PORTUGUESE, use_bert=use_bert
            )
        else:
            logger.info('Generating text, using model: {}'.format(model_file_name))
            words, words_probs = generate(model_file_name, corpus, device, model=model)
            logger.info('Generated text: {}'.format((' ').join(words)))


if __name__ == '__main__':
    main()
