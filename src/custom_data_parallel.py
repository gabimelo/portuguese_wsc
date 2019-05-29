import torch.nn as nn

from src.parallel import DataParallelModel


class CustomDataParallel(nn.Module):
    def __init__(self, model):
        super(CustomDataParallel, self).__init__()
        self.model = DataParallelModel(model)
#         .cuda()

    def forward(self, input, hidden):
        return self.model(input, hidden)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)
