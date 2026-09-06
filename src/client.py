"""Client-side model ownership and fixed-step local optimization."""

from copy import deepcopy

from typing import Dict, Optional

import torch

import torch.nn as nn

from torch.utils.data import DataLoader

from src.sparse_sgd import SparseSGDM


class FederatedClient:

    """A single federated learning client.

    Supports:

        - Standard SGD

        - SparseSGDM with a gradient mask

    Client models are stored on CPU and moved to the training

    device only while that client is performing local training. Its optimizer

    state persists across rounds, allowing momentum to accumulate locally.

    """

    def __init__(

        self,

        client_id: int,

        model: nn.Module,

        train_loader: DataLoader,

        device: torch.device,

        learning_rate: float = 0.01,

        momentum: float = 0.9,

        weight_decay: float = 0.0,

        gradient_mask: Optional[Dict[str, torch.Tensor]] = None,

    ) -> None:

        # Store this client's local training configuration.
        self.client_id = client_id

        self.device = device

        self.train_loader = train_loader

        self.learning_rate = learning_rate

        self.momentum = momentum

        self.weight_decay = weight_decay

        # Keep inactive client models on CPU.

        # deepcopy gives every client its own independent model.
        self.model = deepcopy(model).cpu()

        # Loss function used for CIFAR-100 classification.
        self.criterion = nn.CrossEntropyLoss()

        # Keep inactive masks on CPU.

        if gradient_mask is not None:

            # Store an independent CPU copy of the mask.
            self.gradient_mask = {

                name: mask.detach().cpu().clone()

                for name, mask in gradient_mask.items()

            }

        else:

            self.gradient_mask = None

        # Choose standard SGD or SparseSGDM based on the mask.
        self.optimizer = self._build_optimizer()

    def _build_optimizer(

        self,

    ) -> torch.optim.Optimizer:

        """

        Build SGD or SparseSGDM depending on whether

        a gradient mask exists.

        """

        # No mask means standard local SGDM training.
        if self.gradient_mask is None:

            return torch.optim.SGD(

                filter(

                    lambda parameter: parameter.requires_grad,

                    self.model.parameters(),

                ),

                lr=self.learning_rate,

                momentum=self.momentum,

                weight_decay=self.weight_decay,

            )

        # With a mask, only the allowed parameter updates are applied.
        return SparseSGDM(

            named_parameters=self.model.named_parameters(),

            gradient_mask=self.gradient_mask,

            lr=self.learning_rate,

            momentum=self.momentum,

            weight_decay=self.weight_decay,

        )

    def _move_optimizer_state(

        self,

        device: torch.device,

    ) -> None:

        """

        Move optimizer state tensors to the requested device.

        """

        # This also moves persistent states such as momentum buffers.
        for state in self.optimizer.state.values():

            for key, value in state.items():

                if torch.is_tensor(value):

                    state[key] = value.to(device)

    def _move_gradient_mask(

        self,

        device: torch.device,

    ) -> None:

        """

        Move both the client's gradient mask and the

        SparseSGDM optimizer's internal mask to the

        requested device.

        """

        if self.gradient_mask is None:

            return

        # Move every mask tensor to the same device as the model.
        self.gradient_mask = {

            name: mask.to(device)

            for name, mask in self.gradient_mask.items()

        }

        # SparseSGDM keeps its own internal mask.

        # Synchronize it with the client's moved mask.

        if isinstance(self.optimizer, SparseSGDM):

            self.optimizer.set_gradient_mask(

                self.gradient_mask

            )

    def _move_to_training_device(self) -> None:

        """

        Move this client's model, optimizer state,

        and gradient mask to the training device.

        """

        # Move model first so SparseSGDM can place its

        # internal mask on the same device as parameters.

        self.model.to(self.device)

        self._move_optimizer_state(

            self.device

        )

        self._move_gradient_mask(

            self.device

        )

    def _move_to_cpu(self) -> None:

        """

        Move this client's persistent state back to CPU

        after local training.

        """

        self.optimizer.zero_grad(

            set_to_none=True

        )

        # Move the model first.

        self.model.cpu()

        # Keep the optimizer state across rounds, but store it on CPU.
        self._move_optimizer_state(

            torch.device("cpu")

        )

        # This also synchronizes SparseSGDM's internal mask.

        self._move_gradient_mask(

            torch.device("cpu")

        )

        # Release unused cached GPU memory.
        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    def set_weights(

        self,

        global_model: nn.Module,

    ) -> None:

        """

        Receive the latest global model weights from the server.

        The client model remains on CPU.

        """

        # Synchronize this client with the latest global model.
        self.model.load_state_dict(

            global_model.state_dict()

        )

    def set_gradient_mask(

        self,

        gradient_mask: Optional[

            Dict[str, torch.Tensor]

        ],

    ) -> None:

        """

        Set or replace the client's gradient mask.

        Passing None switches the client back to

        standard SGD.

        """

        if gradient_mask is None:

            self.gradient_mask = None

        else:

            # Store an independent CPU copy of the new mask.
            self.gradient_mask = {

                name: mask.detach().cpu().clone()

                for name, mask in gradient_mask.items()

            }

        # Rebuild the optimizer because the optimizer type may change.
        self.optimizer = self._build_optimizer()

    def train(

        self,

        local_steps: int = 4,

    ) -> Dict[str, float]:

        """

        Perform a fixed number of local optimization steps.

        local_steps corresponds to J in the project specification.

        One local step means one mini-batch update.

        Only the currently training client occupies GPU memory. If the loader

        is exhausted before J steps, iteration restarts so exactly J updates

        are still performed.

        """

        if local_steps <= 0:

            raise ValueError(

                "local_steps must be greater than zero."

            )

        # Move only the currently selected client to the training device.
        self._move_to_training_device()

        try:

            self.model.train()

            # Variables used to calculate local training metrics.
            total_loss = 0.0

            total_correct = 0

            total_samples = 0

            completed_steps = 0

            data_iterator = iter(

                self.train_loader

            )

            # Perform exactly J mini-batch optimization steps.
            while completed_steps < local_steps:

                try:

                    inputs, targets = next(

                        data_iterator

                    )

                except StopIteration:

                    # Restart the loader if it ends before J steps are completed.
                    data_iterator = iter(

                        self.train_loader

                    )

                    inputs, targets = next(

                        data_iterator

                    )

                inputs = inputs.to(

                    self.device,

                    non_blocking=True,

                )

                targets = targets.to(

                    self.device,

                    non_blocking=True,

                )

                # Clear gradients from the previous local step.
                self.optimizer.zero_grad(

                    set_to_none=True

                )

                # Forward pass.
                outputs = self.model(

                    inputs

                )

                # Compute classification loss.
                loss = self.criterion(

                    outputs,

                    targets,

                )

                # Compute gradients.
                loss.backward()

                # Update model parameters using SGD or SparseSGDM.
                self.optimizer.step()

                batch_size = targets.size(0)

                total_loss += (

                    loss.item()

                    * batch_size

                )

                predictions = outputs.argmax(

                    dim=1

                )

                total_correct += (

                    predictions

                    == targets

                ).sum().item()

                total_samples += batch_size

                completed_steps += 1

            # Compute the client's average local loss and accuracy.
            average_loss = (

                total_loss

                / total_samples

            )

            accuracy = (

                total_correct

                / total_samples

            )

            # Return statistics from this client's local training.
            metrics = {

                "loss": average_loss,

                "accuracy": accuracy,

                "steps": completed_steps,

                "samples": total_samples,

                "optimizer": (

                    "SparseSGDM"

                    if self.gradient_mask is not None

                    else "SGD"

                ),

            }

        finally:

            # Always release this client's GPU memory,

            # even if local training fails.

            self._move_to_cpu()

        return metrics

    def get_model(self) -> nn.Module:

        """

        Return the locally trained client model.

        Client models are normally stored on CPU.

        """

        return self.model