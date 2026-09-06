"""Server-side distribution and FedAvg state management."""

from typing import List

import torch
import torch.nn as nn

from src.aggregation import fedavg, update_global_model


class FederatedServer:
    """
    Federated learning server that maintains the global model and aggregates
    client updates using FedAvg.
    """

    def __init__(
        self,
        global_model: nn.Module,
        device: torch.device,
    ) -> None:

        self.global_model = global_model.to(device)
        self.device = device

    def distribute(self, clients) -> None:
        """
        Broadcast the current global state to the selected clients.
        """

        for client in clients:
            client.set_weights(self.global_model)

    def aggregate(
        self,
        client_models: List[nn.Module],
    ) -> None:
        """
        Average selected client states and install the result globally.
        """

        averaged_state = fedavg(client_models)

        update_global_model(
            self.global_model,
            averaged_state,
        )

    def get_global_model(self) -> nn.Module:
        """
        Return the updated global model.
        """

        return self.global_model