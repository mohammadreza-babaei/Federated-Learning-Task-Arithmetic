from typing import Dict

import torch
import torch.nn as nn


def validate_sparsity_ratio(
    sparsity_ratio: float,
) -> None:
    """
    Validate the requested sparsity ratio.

    sparsity_ratio is the fraction of trainable parameters
    that will remain active in the gradient mask.
    """

    if not 0.0 < sparsity_ratio <= 1.0:
        raise ValueError(
            "sparsity_ratio must be greater than 0 "
            "and at most 1."
        )


def flatten_score_dictionary(
    scores: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Flatten all parameter score tensors into one vector.
    """

    if not scores:
        raise ValueError(
            "scores cannot be empty."
        )

    return torch.cat(
        [
            score.detach().reshape(-1)
            for score in scores.values()
        ]
    )


def build_least_sensitive_mask(
    sensitivity_scores: Dict[str, torch.Tensor],
    sparsity_ratio: float,
) -> Dict[str, torch.Tensor]:
    """
    Build a binary mask that activates the least-sensitive parameters.

    The fraction of parameters kept active is controlled by
    sparsity_ratio.
    """

    validate_sparsity_ratio(sparsity_ratio)

    flattened_scores = flatten_score_dictionary(
        sensitivity_scores
    )

    total_parameters = flattened_scores.numel()

    active_parameters = max(
        1,
        int(total_parameters * sparsity_ratio),
    )

    threshold = torch.kthvalue(
        flattened_scores,
        k=active_parameters,
    ).values

    mask = {}

    for name, score in sensitivity_scores.items():
        mask[name] = (
            score <= threshold
        ).to(
            dtype=score.dtype,
            device=score.device,
        )

    return mask


def build_most_sensitive_mask(
    sensitivity_scores: Dict[str, torch.Tensor],
    sparsity_ratio: float,
) -> Dict[str, torch.Tensor]:
    """
    Build a binary mask that activates the most-sensitive parameters.
    """

    validate_sparsity_ratio(sparsity_ratio)

    flattened_scores = flatten_score_dictionary(
        sensitivity_scores
    )

    total_parameters = flattened_scores.numel()

    active_parameters = max(
        1,
        int(total_parameters * sparsity_ratio),
    )

    descending_scores = -flattened_scores

    threshold = torch.kthvalue(
        descending_scores,
        k=active_parameters,
    ).values

    mask = {}

    for name, score in sensitivity_scores.items():
        mask[name] = (
            -score <= threshold
        ).to(
            dtype=score.dtype,
            device=score.device,
        )

    return mask


def build_lowest_magnitude_mask(
    model: nn.Module,
    sparsity_ratio: float,
) -> Dict[str, torch.Tensor]:
    """
    Build a mask that activates the lowest-magnitude parameters.
    """

    validate_sparsity_ratio(sparsity_ratio)

    parameter_scores = {
        name: parameter.detach().abs()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    return build_least_sensitive_mask(
        sensitivity_scores=parameter_scores,
        sparsity_ratio=sparsity_ratio,
    )


def build_highest_magnitude_mask(
    model: nn.Module,
    sparsity_ratio: float,
) -> Dict[str, torch.Tensor]:
    """
    Build a mask that activates the highest-magnitude parameters.
    """

    validate_sparsity_ratio(sparsity_ratio)

    parameter_scores = {
        name: parameter.detach().abs()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    return build_most_sensitive_mask(
        sensitivity_scores=parameter_scores,
        sparsity_ratio=sparsity_ratio,
    )


def build_random_mask(
    model: nn.Module,
    sparsity_ratio: float,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Build a random binary mask over trainable parameters.
    """

    validate_sparsity_ratio(sparsity_ratio)

    trainable_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    if not trainable_parameters:
        raise ValueError(
            "The model has no trainable parameters."
        )

    total_parameters = sum(
        parameter.numel()
        for parameter in trainable_parameters.values()
    )

    active_parameters = max(
        1,
        int(total_parameters * sparsity_ratio),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    random_values = torch.rand(
        total_parameters,
        generator=generator,
    )

    selected_indices = torch.topk(
        random_values,
        k=active_parameters,
        largest=False,
    ).indices

    flat_mask = torch.zeros(
        total_parameters,
        dtype=torch.float32,
    )

    flat_mask[selected_indices] = 1.0

    masks = {}

    offset = 0

    for name, parameter in trainable_parameters.items():
        parameter_size = parameter.numel()

        masks[name] = (
            flat_mask[
                offset:offset + parameter_size
            ]
            .reshape_as(parameter)
            .to(parameter.device)
        )

        offset += parameter_size

    return masks


def calculate_mask_statistics(
    gradient_mask: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Calculate the number and fraction of active mask entries.
    """

    if not gradient_mask:
        raise ValueError(
            "gradient_mask cannot be empty."
        )

    total_entries = sum(
        mask.numel()
        for mask in gradient_mask.values()
    )

    active_entries = sum(
        int(mask.sum().item())
        for mask in gradient_mask.values()
    )

    return {
        "total_entries": total_entries,
        "active_entries": active_entries,
        "active_ratio": active_entries / total_entries,
        "zero_ratio": 1.0 - (
            active_entries / total_entries
        ),
    }