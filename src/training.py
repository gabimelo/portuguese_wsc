import datetime
import time
import math

import torch

from src.consts import (
    BATCH_SIZE, SEQUENCE_LENGTH, EVAL_BATCH_SIZE, INITIAL_LEARNING_RATE, EPOCHS, GRADIENT_CLIPPING,
    LOG_INTERVAL, MODEL_FILE_NAME
)
from src.logger import Logger
from src.utils import batchify, get_batch, repackage_hidden

logger = Logger()


def train(model, corpus, criterion, device):
    timestamp = datetime.datetime.now()
    
    # Loop over epochs.
    lr = INITIAL_LEARNING_RATE
    best_val_loss = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, EPOCHS + 1):
            epoch_start_time = time.time()

            train_one_epoch(model, corpus, criterion, lr, epoch, device)

            val_loss = evaluate(model, corpus, criterion, device)

            logger.info('-' * 89)
            logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                   val_loss, math.exp(val_loss)))
            logger.info('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(MODEL_FILE_NAME.format(timestamp), 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        logger.info('-' * 89)
        logger.info('Exiting from training early')

    test_data = batchify(corpus.test, EVAL_BATCH_SIZE, device)
    get_training_results(model, corpus, criterion, device, timestamp)


def evaluate(model, corpus, criterion, device, use_test_data=False):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(EVAL_BATCH_SIZE)
    if not use_test_data:
        full_data = batchify(corpus.valid, EVAL_BATCH_SIZE, device)
    else:
        full_data = batchify(corpus.test, EVAL_BATCH_SIZE, device)
    with torch.no_grad():
        for i in range(0, full_data.size(0) - 1, SEQUENCE_LENGTH):
            data, targets = get_batch(full_data, i)
#             output, hidden = model(data, hidden)
            output, hidden = model(data.permute(1, 0), (hidden[0].permute(1, 0, 2).contiguous(),
                                                        hidden[1].permute(1, 0, 2).contiguous()))
            hidden = (hidden[0].permute(1, 0, 2).contiguous(), hidden[1].permute(1, 0, 2).contiguous())
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data) - 1)


def train_one_epoch(model, corpus, criterion, lr, epoch, device):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(BATCH_SIZE)
    train_data = batchify(corpus.train, BATCH_SIZE, device)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, SEQUENCE_LENGTH)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
#         output, hidden = model(data, hidden)
        output, hidden = model(data.permute(1, 0), (hidden[0].permute(1, 0, 2).contiguous(),
                                                    hidden[1].permute(1, 0, 2).contiguous()))
        hidden = (hidden[0].permute(1, 0, 2).contiguous(), hidden[1].permute(1, 0, 2).contiguous())
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
                        'loss {:5.2f} | ppl {:8.2f}'.format(epoch, batch, len(train_data) // SEQUENCE_LENGTH, lr,
                                                            elapsed * 1000 / LOG_INTERVAL, cur_loss,
                                                            math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def get_training_results(model, corpus, criterion, device, timestamp):
    # Load the best saved model.
    with open(MODEL_FILE_NAME.format(timestamp), 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(model, corpus, criterion, device, use_test_data=True)
    logger.info('=' * 89)
    logger.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    logger.info('=' * 89)
