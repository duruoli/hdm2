import numpy as np
import torch


def has_nan(x):
    return torch.any(torch.isnan(x))


def has_nan_cpu(x):
    return has_nan(x).cpu().numpy()


def np_has_nan(x):
    return np.any(np.isnan(x))


def grad_has_nan(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    if len(parameters) == 0:
        return torch.zeros(1).mean()
    device = parameters[0].grad.device
    g_has_nan = torch.any(torch.stack([
        torch.any(torch.isnan(p.grad.detach())).to(device) for p in parameters]))
    return g_has_nan


def param_has_nan(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.data is not None, parameters))
    if len(parameters) == 0:
        return torch.zeros(1).mean()
    device = parameters[0].data.device
    p_has_nan = torch.any(torch.stack([
        torch.any(torch.isnan(p.data.detach())).to(device) for p in parameters]))
    return p_has_nan
