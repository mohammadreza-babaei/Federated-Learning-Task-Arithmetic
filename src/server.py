from typing import List

import torch
import torch.nn as nn

from src.aggregation import fedavg, update_global_model


class FederatedServer:
    """
    Federated learning server.
    Responsible for maintaining the global model and aggregating
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
        Send the current global model to every client.
        """

        for client in clients:
            client.set_weights(self.global_model)

    def aggregate(
        self,
        client_models: List[nn.Module],
    ) -> None:
        """
        Aggregate client models using FedAvg.
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