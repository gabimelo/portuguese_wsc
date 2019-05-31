import datetime
import time
import math

from tqdm import tqdm
import torch

from src.consts import (
    BATCH_SIZE, SEQUENCE_LENGTH, EVAL_BATCH_SIZE, INITIAL_LEARNING_RATE, EPOCHS, GRADIENT_CLIPPING,
    LOG_INTERVAL, MODEL_FILE_NAME, MODEL_RESULTS_FILE_NAME
)
from src.logger import Logger
from src.utils import (
    batchify, get_batch, repackage_hidden, permute_for_parallelization, get_results_from_data_parallelized_forward
)

logger = Logger()


def train(model, corpus, criterion, optimizer, device, use_data_paralellization):
    timestamp = datetime.datetime.now()

    # Loop over epochs.
    lr = INITIAL_LEARNING_RATE
    best_val_loss = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, EPOCHS + 1):
            epoch_start_time = time.time()

            train_one_epoch(model, corpus, criterion, optimizer, lr, epoch, device, use_data_paralellization)

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

    test_loss = get_training_results(model, corpus, criterion, device, timestamp)

    if best_val_loss is None:
        best_val_loss = 0
    with open(MODEL_RESULTS_FILE_NAME.format(timestamp), 'w') as f:
        f.write('final lr: {}\ntest loss: {:5.2f}\ntest ppl: {:8.2f}\nbest val loss: {:5.2f}\n'
                'epochs: {}\ntime to run: {:5.2f}s'
                .format(lr, test_loss, math.exp(test_loss), best_val_loss, epoch, (time.time() - epoch_start_time)))


def evaluate(model, corpus, criterion, device, use_test_data=False):
    # Turn on evaluation mode which disables dropout.
    logger.info('-' * 89)
    logger.info('Running eval')
    logger.info('-' * 89)

    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(EVAL_BATCH_SIZE)
    if not use_test_data:
        full_data = batchify(corpus.valid, EVAL_BATCH_SIZE, device)
    else:
        full_data = batchify(corpus.test, EVAL_BATCH_SIZE, device)
    with torch.no_grad():
        for batch, i in tqdm(enumerate(range(0, full_data.size(0) - 1, SEQUENCE_LENGTH))):
            data, targets = get_batch(full_data, i)
            output, hidden = model(data, hidden)
            total_loss += len(data) * criterion(output.view(-1, ntokens), targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(full_data) - 1)


def train_one_epoch(model, corpus, criterion, optimizer, lr, epoch, device, use_data_paralellization):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    batch_size = BATCH_SIZE

    if use_data_paralellization:
        batch_size *= torch.cuda.device_count()

    hidden = model.init_hidden(batch_size)
    train_data = batchify(corpus.train, batch_size, device)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, SEQUENCE_LENGTH)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)

        if optimizer is None:
            model.zero_grad()
        else:
            optimizer.zero_grad()

        if use_data_paralellization:
            # code seems to be slightly (~12ms/batch, with batch_size=40 doing it this way instead
            # of setting dim=1 when instantiating DataParallelModel)
            hidden, data = permute_for_parallelization(hidden, data)

            results = model(data, hidden)

            outputs, hidden = get_results_from_data_parallelized_forward(results, device)
            hidden = permute_for_parallelization(hidden)
        else:
            outputs, hidden = model(data.to(device), hidden)
            targets = targets.to(device)

        loss = criterion(outputs, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)

        if optimizer is None:
            for p in model.parameters():
                p.data.add_(-lr, p.grad.data)
        else:
            optimizer.step()

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

    return test_loss
