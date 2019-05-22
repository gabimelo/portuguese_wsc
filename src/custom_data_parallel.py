import torch.nn as nn


class CustomDataParallel(nn.Module):
    def __init__(self, model):
        super(CustomDataParallel, self).__init__()
        self.model = nn.DataParallel(model).cuda()

    def forward(self, *input):
        return self.model(*input)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)
