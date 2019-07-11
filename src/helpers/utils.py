import os
import glob

import torch

from src.consts import MODEL_FILE_NAME


def get_latest_model_file():
    models_dir = ('/').join(MODEL_FILE_NAME.split('/')[:-1]) + '/*.pt'
    list_of_model_files = glob.glob(models_dir)
    latest_file_path = max(list_of_model_files, key=os.path.getctime)

    return latest_file_path


def summary(model, criterion=None):
    print(model, end="\n\n")

    for key, value in model.state_dict().items():
        print(key, value.size())

    params = list(model.parameters())
    if criterion is not None:
        params += list(criterion.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())

    print("\nTotal Parameters: {:,}".format(total_params))


def check_cuda_mem(device):
    print('Max mem', torch.cuda.max_memory_allocated(device=device))
    print('Mem', torch.cuda.memory_allocated(device=device))
