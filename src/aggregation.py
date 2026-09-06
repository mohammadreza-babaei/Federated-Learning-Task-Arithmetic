"""Model-state aggregation utilities used by the federated server."""

from collections import OrderedDict
# OrderedDict stores model parameters while preserving their original order.

from typing import List
# List is used for type hints because FedAvg receives multiple client models.

import torch
# PyTorch is used for tensor operations during model aggregation.

import torch.nn as nn
# nn.Module represents the neural-network models used by the clients and server.


def fedavg(models: List[nn.Module]) -> OrderedDict:

    """Compute the unweighted FedAvg mean of client state dictionaries.

    Args:

        models: List of client models after local training.

    Returns:

        An ``OrderedDict`` compatible with ``load_state_dict``.

    Every state entry, including buffers, is averaged so the server receives a

    complete model state after each communication round.

    """

    # FedAvg requires at least one locally trained client model.
    if len(models) == 0:

        raise ValueError("At least one client model is required.")

    # Create an empty ordered dictionary to store the averaged global model state.
    averaged_state = OrderedDict()

    # Extract the complete state dictionary from every client model.
    # A state_dict contains model parameters and registered buffers.
    state_dicts = [model.state_dict() for model in models]

    # Iterate over every parameter or buffer contained in the model state.
    for key in state_dicts[0].keys():

        # Create a zero tensor with the same shape and type as the current state entry.
        # This tensor will accumulate the values received from all clients.
        averaged_state[key] = torch.zeros_like(state_dicts[0][key])

        # Add the corresponding parameter or buffer from every client model.
        for state_dict in state_dicts:

            averaged_state[key] += state_dict[key]

        # Divide the accumulated value by the number of participating clients.
        # This computes the unweighted arithmetic mean used by this FedAvg implementation.
        averaged_state[key] /= len(state_dicts)

    # Return the complete averaged model state to the federated server.
    return averaged_state


def update_global_model(

    global_model: nn.Module,

    averaged_state: OrderedDict,

) -> nn.Module:

    """Replace the server model state with an already-averaged state."""

    # Load the aggregated client parameters and buffers into the server's global model.
    global_model.load_state_dict(averaged_state)

    # Return the updated global model for the next federated communication round.
    return global_model