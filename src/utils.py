import os
import glob

import torch

from src.consts import SEQUENCE_LENGTH, MODEL_FILE_NAME


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
def batchify(data, batch_size, device):
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


def get_latest_model_file():
    models_dir = ('/').join(MODEL_FILE_NAME.split('/')[:-1]) + '/*'
    list_of_model_files = glob.glob(models_dir)
    latest_file_path = max(list_of_model_files, key=os.path.getctime)

    return latest_file_path


def summary(model, criterion):
    print(model, end="\n\n")

    for key, value in model.state_dict().items():
        print(key, value.size())

    params = list(model.parameters()) + list(criterion.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())

    print("\nTotal Parameters: {:,}".format(total_params))


def check_cuda_mem(device):
    print('Max mem', torch.cuda.max_memory_allocated(device=device))
    print('Mem', torch.cuda.memory_allocated(device=device))


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
