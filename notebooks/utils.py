import torch


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
