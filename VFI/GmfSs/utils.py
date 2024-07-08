import torch

ones_cache = {}


def get_ones_tensor(tensor: torch.Tensor):
    k = (str(tensor.device), str(tensor.size()))
    if k in ones_cache:
        return ones_cache[k]
    ones_cache[k] = torch.ones(tensor.size(), requires_grad=False, dtype=tensor.dtype).to(tensor.device)
    return ones_cache[k]


def get_ones_tensor_size(size: tuple, device, dtype: torch.dtype):
    k = (str(device), str(size))
    if k in ones_cache:
        return ones_cache[k]
    ones_cache[k] = torch.ones(size, requires_grad=False, dtype=dtype).to(device)
    return ones_cache[k]
