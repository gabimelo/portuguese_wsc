import torch.nn as nn

from src.parallel import DataParallelModel


class CustomDataParallel(nn.Module):
    def __init__(self, model, device_ids=None, output_device=None, dim=0):
        super(CustomDataParallel, self).__init__()
        self.model = DataParallelModel(model, device_ids, output_device, dim)

    def forward(self, *input):
        return self.model(*input)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)
