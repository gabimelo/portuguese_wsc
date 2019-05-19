# # coding: utf-8
# import argparse
import time
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

logger = Logger()


def main():
    # Set the random seed manually for reproducibility.
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        if not USE_CUDA:
            logger.info("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda" if USE_CUDA else "cpu")

    corpus = Corpus()
    corpus.add_corpus_data(PROCESSED_DATA_DIR_NAME)

    train_data = batchify(corpus.train, BATCH_SIZE, device)
    val_data = batchify(corpus.valid, EVAL_BATCH_SIZE, device)
    test_data = batchify(corpus.test, EVAL_BATCH_SIZE, device)

    ntokens = len(corpus.dictionary)
    model = RNNModel(MODEL_TYPE, ntokens, EMBEDDINGS_SIZE, HIDDEN_UNIT_COUNT, LAYER_COUNT, DROPOUT_PROB,
                     TIED).to(device)
    criterion = nn.CrossEntropyLoss()

    train(train_data, val_data, model, corpus, criterion)

    get_training_results(test_data)


def train(train_data, val_data, model, corpus, criterion):
    # Loop over epochs.
    lr = INITIAL_LEARNING_RATE
    best_val_loss = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, EPOCHS+1):
            epoch_start_time = time.time()
            train_one_epoch(model, corpus, train_data, criterion, lr, epoch)
            val_loss = evaluate(val_data)
            logger.info('-' * 89)
            logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            logger.info('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(MODEL_FILE_NAME, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        logger.info('-' * 89)
        logger.info('Exiting from training early')


def get_training_results(test_data):
    # Load the best saved model.
    with open(MODEL_FILE_NAME, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(test_data)
    logger.info('=' * 89)
    logger.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    logger.info('=' * 89)


# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.
def batchify(data, batch_size, device):
    # Work out how cleanly we can divide the dataset into batch_size parts.
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length SEQUENCE_LENGTH.
# If source is equal to the example output of the batchify function, with
# a SEQUENCE_LENGTH of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.
def get_batch(source, i):
    seq_len = min(SEQUENCE_LENGTH, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source, model, corpus, criterion):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(EVAL_BATCH_SIZE)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, SEQUENCE_LENGTH):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)


def train_one_epoch(model, corpus, train_data, criterion, lr, epoch):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(BATCH_SIZE)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, SEQUENCE_LENGTH)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % LOG_INTERVAL == 0 and batch > 0:
            cur_loss = total_loss / LOG_INTERVAL
            elapsed = time.time() - start_time
            logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // SEQUENCE_LENGTH, lr,
                elapsed * 1000 / LOG_INTERVAL, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


if __name__ == '__main__':
    main()
