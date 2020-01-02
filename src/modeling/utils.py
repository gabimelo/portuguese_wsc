import torch

from src.helpers.consts import SEQUENCE_LENGTH


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


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
def batchify(data, batch_size):
    # Work out how cleanly we can divide the dataset into batch_size parts.
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data


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
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def permute_for_parallelization(hidden, data=None):
    hidden = (hidden[0].permute(1, 0, 2).contiguous(),
              hidden[1].permute(1, 0, 2).contiguous())
    if data is None:
        return hidden
    data = data.permute(1, 0)

    return hidden, data


def get_results_from_data_parallelized_forward(results, device):
    outputs = []
    hidden_0 = torch.Tensor().to(device)
    hidden_1 = torch.Tensor().to(device)

    for result in results:
        outputs.append(result[0])
        hidden_0 = torch.cat((hidden_0, result[1][0]))
        hidden_1 = torch.cat((hidden_1, result[1][1]))
        del result

    return tuple(outputs), (hidden_0, hidden_1)
