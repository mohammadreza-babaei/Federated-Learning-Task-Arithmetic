"""DINO ViT-S/16 feature extraction and CIFAR-100 classification head."""

from typing import Any

import torch
import torch.nn as nn


class DinoViTS16CIFAR100(nn.Module):
    """Wrap the DINO backbone with a CIFAR-100 classification head.

    The wrapper accepts common DINO output formats and normalizes them to one
    feature vector per image before classification.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 100,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError("num_classes must be greater than zero.")

        self.num_classes = num_classes
        self.backbone = backbone
        self.backbone_frozen = False

        self.backbone_dim = getattr(self.backbone, "embed_dim", 384)
        self.classifier_head = nn.Linear(
            in_features=self.backbone_dim,
            out_features=num_classes,
        )

        if freeze_backbone:
            self.freeze_backbone()

    def _extract_features(self, output: Any) -> torch.Tensor:
        """Extract and validate one CLS-style feature vector per image."""

        if isinstance(output, dict):
            if "x_norm_clstoken" in output:
                output = output["x_norm_clstoken"]
            elif "x" in output:
                output = output["x"]
            elif "features" in output:
                output = output["features"]
            elif "out" in output:
                output = output["out"]
            else:
                raise RuntimeError(
                    "Unsupported dictionary output from the DINO backbone. "
                    f"Available keys: {list(output.keys())}"
                )

        if isinstance(output, (tuple, list)):
            if len(output) == 0:
                raise RuntimeError("The DINO backbone returned an empty output.")
            output = output[0]

        if not isinstance(output, torch.Tensor):
            raise RuntimeError(
                "The DINO backbone output must be a torch.Tensor, "
                f"but received {type(output).__name__}."
            )

        # Some ViT implementations return all tokens:
        # [batch_size, number_of_tokens, embedding_dimension].
        # In that case, token zero is the CLS token.
        if output.ndim == 3:
            output = output[:, 0]

        if output.ndim == 1:
            output = output.unsqueeze(0)

        if output.ndim != 2:
            raise RuntimeError(
                "Expected DINO features with shape [batch_size, embedding_dim], "
                f"but received {tuple(output.shape)}."
            )

        if output.shape[1] != self.backbone_dim:
            raise RuntimeError(
                f"Expected feature dimension {self.backbone_dim}, "
                f"but received {output.shape[1]}."
            )

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return unnormalized logits for the configured CIFAR-100 classes."""

        # For the official facebookresearch/dino implementation,
        # forward() normally returns the CLS-token feature vector.
        output = self.backbone(x)
        features = self._extract_features(output)

        return self.classifier_head(features)

    def freeze_backbone(self) -> None:
        """Freeze the DINO backbone while keeping the classifier trainable."""

        self.backbone_frozen = True

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for parameter in self.classifier_head.parameters():
            parameter.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Enable training for all DINO backbone parameters."""

        self.backbone_frozen = False

        for parameter in self.backbone.parameters():
            parameter.requires_grad = True


def load_official_dino_vits16(pretrained: bool = True) -> nn.Module:
    """Load the official DINO ViT-S/16 architecture through ``torch.hub``."""

    try:
        backbone = torch.hub.load(
            "facebookresearch/dino:main",
            "dino_vits16",
            pretrained=pretrained,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the official DINO ViT-S/16 model from "
            "facebookresearch/dino. Check the internet connection and "
            "the local torch.hub cache."
        ) from exc

    if not isinstance(backbone, nn.Module):
        raise RuntimeError(
            "torch.hub did not return a valid PyTorch model."
        )

    return backbone


def build_dino_vits16_cifar100(
    num_classes: int = 100,
    freeze_backbone: bool = False,
    pretrained: bool = True,
) -> DinoViTS16CIFAR100:
    """Load DINO and return the CIFAR-100 classifier wrapper."""

    backbone = load_official_dino_vits16(pretrained=pretrained)

    return DinoViTS16CIFAR100(
        backbone=backbone,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
    )