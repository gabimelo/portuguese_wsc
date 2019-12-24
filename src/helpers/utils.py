import os
import glob

import torch

from src.helpers.consts import MODEL_FILE_NAME
from src.helpers.logger import Logger

logger = Logger()


def get_latest_model_file():
    models_dir = ('/').join(MODEL_FILE_NAME.split('/')[:-1]) + '/*.pt'
    list_of_model_files = glob.glob(models_dir)
    latest_file_path = max(list_of_model_files, key=os.path.getctime)

    return latest_file_path


def summary(model, criterion=None):
    logger.info(model, end="\n\n")

    for key, value in model.state_dict().items():
        logger.info(key, value.size())

    params = list(model.parameters())
    if criterion is not None:
        params += list(criterion.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())

    logger.info("\nTotal Parameters: {:,}".format(total_params))


def check_cuda_mem(device):
    logger.info('Max mem', torch.cuda.max_memory_allocated(device=device))
    logger.info('Mem', torch.cuda.memory_allocated(device=device))


def log_loaded_model_info(model_file_name, device):
    with open(model_file_name, 'rb') as f:
        model = torch.load(f).to(device)
    summary(model)

    logger.info('Model training results:')
    with open(model_file_name.replace('model-', 'model-results-').replace('.pt', '.txt'), 'r') as file:
        model_training_results = file.read()
        logger.info(model_training_results)
