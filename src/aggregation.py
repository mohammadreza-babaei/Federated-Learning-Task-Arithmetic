from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn


def fedavg(models: List[nn.Module]) -> OrderedDict:
    """
    Federated Averaging (FedAvg).

    Args:
        models: List of client models after local training.

    Returns:
        Averaged state_dict.
    """

    if len(models) == 0:
        raise ValueError("At least one client model is required.")

    averaged_state = OrderedDict()

    state_dicts = [model.state_dict() for model in models]

    for key in state_dicts[0].keys():

        averaged_state[key] = torch.zeros_like(state_dicts[0][key])

        for state_dict in state_dicts:
            averaged_state[key] += state_dict[key]

        averaged_state[key] /= len(state_dicts)

    return averaged_state


def update_global_model(
    global_model: nn.Module,
    averaged_state: OrderedDict,
) -> nn.Module:
    """
    Load the averaged parameters into the global model.
    """

    global_model.load_state_dict(averaged_state)

    return global_model