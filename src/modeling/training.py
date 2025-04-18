from datetime import datetime
import math

from tqdm import tqdm
import torch

from src.helpers.consts import (
    BATCH_SIZE, SEQUENCE_LENGTH, EVAL_BATCH_SIZE, INITIAL_LEARNING_RATE, EPOCHS, GRADIENT_CLIPPING,
    LOG_INTERVAL, MODEL_FILE_NAME, MODEL_RESULTS_FILE_NAME
)
from src.helpers.logger import Logger
from src.modeling.utils import (
    batchify, get_batch, repackage_hidden, permute_for_parallelization, get_results_from_data_parallelized_forward
)

logger = Logger()


def train(model, corpus, criterion, optimizer, device, use_data_paralellization):
    training_start_time = datetime.utcnow()

    # Loop over epochs.
    lr = INITIAL_LEARNING_RATE
    best_val_loss = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, EPOCHS + 1):
            epoch_start_time = datetime.utcnow()

            train_one_epoch(model, corpus, criterion, optimizer, lr, epoch, device, use_data_paralellization)

            val_loss = evaluate(model, corpus, criterion, device, use_data_paralellization=use_data_paralellization)

            logger.info('-' * 89)
            logger.info(
                f'| end of epoch {epoch:3d} | time: {str(datetime.utcnow() - epoch_start_time)}s '
                f'| valid loss {val_loss:5.2f} '
                f'| valid ppl {math.exp(val_loss):8.2f}')
            logger.info('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if best_val_loss is None or val_loss < best_val_loss:
                with open(MODEL_FILE_NAME.format(training_start_time), 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        logger.info('-' * 89)
        logger.info('Exiting from training early')

    test_loss = get_training_results(model, corpus, criterion, device, training_start_time,
                                     use_data_paralellization=use_data_paralellization)

    if best_val_loss is None:
        best_val_loss = 0
    with open(MODEL_RESULTS_FILE_NAME.format(training_start_time), 'w') as f:
        f.write(
            f'final lr: {lr}\ntest loss: {test_loss:5.2f}\ntest ppl: {math.exp(test_loss):8.2f}\n'
            f'best val loss: {best_val_loss:5.2f}\nepochs: {epoch}\n'
            f'time to run: {str(datetime.utcnow() - training_start_time)}s'
        )


def train_one_epoch(model, corpus, criterion, optimizer, lr, epoch, device, use_data_paralellization):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = datetime.utcnow()
    batch_size = BATCH_SIZE

    if use_data_paralellization:
        batch_size *= torch.cuda.device_count()

    hidden = model.init_hidden(batch_size)
    train_data = batchify(corpus.train, batch_size)

    for i in range(0, train_data.size(0) - 1, SEQUENCE_LENGTH):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)

        if optimizer is None:
            model.zero_grad()
        else:
            optimizer.zero_grad()

        if use_data_paralellization:
            # code seems to be slightly faster (~12ms/batch, with batch_size=40) doing it this way instead
            # of setting dim=1 when instantiating DataParallelModel)
            hidden, data = permute_for_parallelization(hidden, data)
            results = model(data, hidden)
            output, hidden = get_results_from_data_parallelized_forward(results, device)
            hidden = permute_for_parallelization(hidden)
        else:
            output, hidden = model(data.to(device), hidden)

        loss = criterion(output, targets.to(device))
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)

        if optimizer is None:
            for p in model.parameters():
                p.data.add_(-lr, p.grad.data)
        else:
            optimizer.step()

        total_loss += loss.item()

        if i % LOG_INTERVAL == 0 and i > 0:
            cur_loss = total_loss / LOG_INTERVAL
            elapsed = datetime.utcnow() - start_time
            try:
                cur_ppl = f'{math.exp(cur_loss):8.2f}'
            except OverflowError:
                cur_ppl = 'OVERFLOW'

            logger.info(
                f'| epoch {epoch:3d} | batch {(i // SEQUENCE_LENGTH):5d} /{(len(train_data) // SEQUENCE_LENGTH):5d} '
                f'| lr {lr:02.2f} | ms/batch {(elapsed.microseconds / LOG_INTERVAL):5.2f} '
                f'| loss {cur_loss:5.2f} | ppl {cur_ppl}'
            )
            total_loss = 0
            start_time = datetime.utcnow()


def evaluate(model, corpus, criterion, device, use_test_data=False, use_train_data=False,
             use_data_paralellization=False):
    logger.info('-' * 89)
    logger.info('Running eval')
    logger.info('-' * 89)

    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.

    if use_train_data:
        batch_size = BATCH_SIZE
        selected_corpus = corpus.train
    elif not use_test_data:
        batch_size = EVAL_BATCH_SIZE
        selected_corpus = corpus.valid
    else:
        batch_size = EVAL_BATCH_SIZE
        selected_corpus = corpus.test

    if use_data_paralellization:
        batch_size *= torch.cuda.device_count()

    hidden = model.init_hidden(batch_size)
    data_source = batchify(selected_corpus, batch_size)

    with torch.no_grad():
        for i in tqdm(range(0, data_source.size(0) - 1, SEQUENCE_LENGTH)):
            data, targets = get_batch(data_source, i)

            if use_data_paralellization:
                hidden, data = permute_for_parallelization(hidden, data)
                results = model(data, hidden)
                output, hidden = get_results_from_data_parallelized_forward(results, device)
                hidden = permute_for_parallelization(hidden)
            else:
                output, hidden = model(data.to(device), hidden)

            total_loss += len(data) * criterion(output, targets.to(device)).item()
            hidden = repackage_hidden(hidden)

    return total_loss / (len(data_source) - 1)


def get_training_results(model, corpus, criterion, device, timestamp, use_data_paralellization=False):
    # Load the best saved model.
    with open(MODEL_FILE_NAME.format(timestamp), 'rb') as f:
        model = torch.load(f)

    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(model, corpus, criterion, device, use_test_data=True,
                         use_data_paralellization=use_data_paralellization)
    logger.info('=' * 89)
    logger.info(f'| End of training | test loss {test_loss:5.2f} | test ppl {math.exp(test_loss):8.2f}')
    logger.info('=' * 89)

    return test_loss
