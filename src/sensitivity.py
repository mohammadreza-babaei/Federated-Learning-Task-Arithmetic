from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def initialize_sensitivity_scores(
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """
    Create one zero-filled sensitivity tensor for each trainable parameter.
    """

    return {
        name: torch.zeros_like(
            parameter,
            device=parameter.device,
        )
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def compute_fisher_sensitivity(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Approximate the diagonal Fisher Information Matrix.

    For each trainable parameter, the sensitivity score is estimated
    by averaging the squared gradients over calibration batches.
    """

    if max_batches is not None and max_batches <= 0:
        raise ValueError(
            "max_batches must be greater than zero or None."
        )

    model = model.to(device)
    model.train()

    sensitivity_scores = initialize_sensitivity_scores(model)

    processed_batches = 0

    for batch_index, (inputs, targets) in enumerate(dataloader):
        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        inputs = inputs.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        model.zero_grad(set_to_none=True)

        outputs = model(inputs)

        loss = criterion(
            outputs,
            targets,
        )

        loss.backward()

        for name, parameter in model.named_parameters():
            if (
                parameter.requires_grad
                and parameter.grad is not None
            ):
                sensitivity_scores[name].add_(
                    parameter.grad.detach().pow(2)
                )

        processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError(
            "No calibration batches were processed."
        )

    for name in sensitivity_scores:
        sensitivity_scores[name].div_(processed_batches)

    return sensitivity_scores


def average_sensitivity_scores(
    sensitivity_rounds: list[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Average sensitivity scores obtained from multiple calibration rounds.
    """

    if not sensitivity_rounds:
        raise ValueError(
            "sensitivity_rounds cannot be empty."
        )

    reference_keys = set(sensitivity_rounds[0].keys())

    for round_scores in sensitivity_rounds[1:]:
        if set(round_scores.keys()) != reference_keys:
            raise ValueError(
                "All sensitivity dictionaries must contain "
                "the same parameter names."
            )

    averaged_scores = {
        name: torch.zeros_like(score)
        for name, score in sensitivity_rounds[0].items()
    }

    for round_scores in sensitivity_rounds:
        for name, score in round_scores.items():
            averaged_scores[name].add_(score)

    number_of_rounds = len(sensitivity_rounds)

    for name in averaged_scores:
        averaged_scores[name].div_(number_of_rounds)

    return averaged_scores