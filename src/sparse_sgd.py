from typing import Dict, Iterable, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer


class SparseSGDM(Optimizer):
    """
    SGD with momentum and a parameter-wise gradient mask.

    Mask value:
        1 -> parameter can be updated
        0 -> parameter update is blocked
    """

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, Tensor]],
        gradient_mask: Dict[str, Tensor],
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ) -> None:

        if lr < 0.0:
            raise ValueError(
                f"Invalid learning rate: {lr}"
            )

        if momentum < 0.0:
            raise ValueError(
                f"Invalid momentum value: {momentum}"
            )

        if weight_decay < 0.0:
            raise ValueError(
                f"Invalid weight decay value: {weight_decay}"
            )

        if dampening < 0.0:
            raise ValueError(
                f"Invalid dampening value: {dampening}"
            )

        if nesterov and (
            momentum <= 0.0
            or dampening != 0.0
        ):
            raise ValueError(
                "Nesterov momentum requires positive momentum "
                "and zero dampening."
            )

        named_parameters = [
            (name, parameter)
            for name, parameter in named_parameters
            if parameter.requires_grad
        ]

        if not named_parameters:
            raise ValueError(
                "No trainable parameters were provided."
            )

        parameter_names = [
            name
            for name, _ in named_parameters
        ]

        missing_masks = [
            name
            for name in parameter_names
            if name not in gradient_mask
        ]

        if missing_masks:
            raise ValueError(
                "Gradient mask is missing parameters: "
                + ", ".join(missing_masks)
            )

        parameters = [
            parameter
            for _, parameter in named_parameters
        ]

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "dampening": dampening,
            "nesterov": nesterov,
        }

        super().__init__(
            parameters,
            defaults,
        )

        self.parameter_names = {
            id(parameter): name
            for name, parameter in named_parameters
        }

        self.gradient_mask = {}

        for name, parameter in named_parameters:
            mask = gradient_mask[name]

            if mask.shape != parameter.shape:
                raise ValueError(
                    f"Mask shape mismatch for {name}: "
                    f"expected {tuple(parameter.shape)}, "
                    f"received {tuple(mask.shape)}."
                )

            self.gradient_mask[name] = mask.detach().to(
                device=parameter.device,
                dtype=parameter.dtype,
            )

    @torch.no_grad()
    def step(
        self,
        closure: Optional[callable] = None,
    ) -> Optional[Tensor]:
        """
        Perform one SparseSGDM optimization step.
        """

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            learning_rate = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                parameter_name = self.parameter_names[
                    id(parameter)
                ]

                mask = self.gradient_mask[
                    parameter_name
                ]

                gradient = parameter.grad.detach()

                if gradient.is_sparse:
                    raise RuntimeError(
                        "SparseSGDM does not support sparse gradients."
                    )

                gradient = gradient.mul(mask)

                if weight_decay != 0.0:
                    gradient = gradient.add(
                        parameter,
                        alpha=weight_decay,
                    )

                    gradient = gradient.mul(mask)

                if momentum != 0.0:
                    state = self.state[parameter]

                    if "momentum_buffer" not in state:
                        momentum_buffer = gradient.clone()
                    else:
                        momentum_buffer = state[
                            "momentum_buffer"
                        ]

                        momentum_buffer.mul_(
                            momentum
                        ).add_(
                            gradient,
                            alpha=1.0 - dampening,
                        )

                    momentum_buffer.mul_(mask)

                    state[
                        "momentum_buffer"
                    ] = momentum_buffer

                    if nesterov:
                        update = gradient.add(
                            momentum_buffer,
                            alpha=momentum,
                        )
                    else:
                        update = momentum_buffer

                else:
                    update = gradient

                update = update.mul(mask)

                parameter.add_(
                    update,
                    alpha=-learning_rate,
                )

        return loss

    def set_gradient_mask(
        self,
        gradient_mask: Dict[str, Tensor],
    ) -> None:
        """
        Replace the current gradient mask.
        """

        for group in self.param_groups:
            for parameter in group["params"]:
                parameter_name = self.parameter_names[
                    id(parameter)
                ]

                if parameter_name not in gradient_mask:
                    raise ValueError(
                        f"Missing mask for parameter: "
                        f"{parameter_name}"
                    )

                mask = gradient_mask[
                    parameter_name
                ]

                if mask.shape != parameter.shape:
                    raise ValueError(
                        f"Mask shape mismatch for "
                        f"{parameter_name}."
                    )

                self.gradient_mask[
                    parameter_name
                ] = mask.detach().to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )