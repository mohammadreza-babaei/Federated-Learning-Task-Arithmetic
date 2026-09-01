"""DINO ViT-S/16 feature extraction and CIFAR-100 classification head."""

from typing import Any

import torch

import torch.nn as nn


# Define a PyTorch model that combines the DINO backbone with a CIFAR-100 classifier.
class DinoViTS16CIFAR100(nn.Module):

    """Wrap the DINO backbone with a CIFAR-100 classification head.

    The wrapper accepts common DINO output formats and normalizes them to one

    feature vector per image before classification.

    """

    # Initialize the DINO backbone and the classification head.
    def __init__(

        self,

        backbone: nn.Module,

        num_classes: int = 100,

        freeze_backbone: bool = False,

    ) -> None:

        # Initialize the parent PyTorch nn.Module class.
        super().__init__()

        # The classifier must contain at least one output class.
        if num_classes <= 0:

            raise ValueError("num_classes must be greater than zero.")

        # Store the number of target classes.
        self.num_classes = num_classes

        # Store the pretrained DINO model used for feature extraction.
        self.backbone = backbone

        # Track whether the backbone parameters are currently frozen.
        self.backbone_frozen = False

        # Read the DINO embedding dimension, using 384 as the default for ViT-S/16.
        self.backbone_dim = getattr(self.backbone, "embed_dim", 384)

        # Create a linear classification layer that maps DINO features to class scores.
        self.classifier_head = nn.Linear(

            in_features=self.backbone_dim,

            out_features=num_classes,

        )

        # Optionally freeze the pretrained DINO backbone.
        if freeze_backbone:

            self.freeze_backbone()

    # Convert different possible DINO outputs into one feature vector per image.
    def _extract_features(self, output: Any) -> torch.Tensor:

        """Extract and validate one CLS-style feature vector per image."""

        # Some DINO implementations return their features inside a dictionary.
        if isinstance(output, dict):

            # Prefer the normalized CLS token when it is directly available.
            if "x_norm_clstoken" in output:

                output = output["x_norm_clstoken"]

            # Otherwise try other commonly used feature keys.
            elif "x" in output:

                output = output["x"]

            elif "features" in output:

                output = output["features"]

            elif "out" in output:

                output = output["out"]

            # Reject dictionary formats that this wrapper does not recognize.
            else:

                raise RuntimeError(

                    "Unsupported dictionary output from the DINO backbone. "

                    f"Available keys: {list(output.keys())}"

                )

        # Some model implementations return multiple outputs as a tuple or list.
        if isinstance(output, (tuple, list)):

            # An empty output cannot contain usable features.
            if len(output) == 0:

                raise RuntimeError("The DINO backbone returned an empty output.")

            # Use the first returned item as the feature representation.
            output = output[0]

        # After handling supported formats, the result must be a PyTorch tensor.
        if not isinstance(output, torch.Tensor):

            raise RuntimeError(

                "The DINO backbone output must be a torch.Tensor, "

                f"but received {type(output).__name__}."

            )

        # Some ViT implementations return all tokens:
        # [batch_size, number_of_tokens, embedding_dimension].
        # In that case, token zero is the CLS token.
        if output.ndim == 3:

            # Select the CLS token as the image-level representation.
            output = output[:, 0]

        # Add a batch dimension when only one feature vector is returned.
        if output.ndim == 1:

            output = output.unsqueeze(0)

        # The classifier expects one feature vector per image.
        if output.ndim != 2:

            raise RuntimeError(

                "Expected DINO features with shape [batch_size, embedding_dim], "

                f"but received {tuple(output.shape)}."

            )

        # Verify that the extracted feature size matches the classifier input size.
        if output.shape[1] != self.backbone_dim:

            raise RuntimeError(

                f"Expected feature dimension {self.backbone_dim}, "

                f"but received {output.shape[1]}."

            )

        # Return the validated image-level feature vectors.
        return output

    # Define how input images pass through the complete model.
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """Return unnormalized logits for the configured CIFAR-100 classes."""

        # For the official facebookresearch/dino implementation,
        # forward() normally returns the CLS-token feature vector.

        # Pass the input images through DINO to extract representations.
        output = self.backbone(x)

        # Convert the backbone output into the expected feature format.
        features = self._extract_features(output)

        # Convert DINO features into CIFAR-100 class logits.
        return self.classifier_head(features)

    # Disable training of the pretrained DINO backbone parameters.
    def freeze_backbone(self) -> None:

        """Freeze the DINO backbone while keeping the classifier trainable."""

        # Record that the backbone is frozen.
        self.backbone_frozen = True

        # Prevent gradients from updating DINO backbone parameters.
        for parameter in self.backbone.parameters():

            parameter.requires_grad = False

        # Keep the classification head trainable.
        for parameter in self.classifier_head.parameters():

            parameter.requires_grad = True

    # Re-enable training of all DINO backbone parameters.
    def unfreeze_backbone(self) -> None:

        """Enable training for all DINO backbone parameters."""

        # Record that the backbone is no longer frozen.
        self.backbone_frozen = False

        # Allow gradients to update all backbone parameters.
        for parameter in self.backbone.parameters():

            parameter.requires_grad = True


# Load the official DINO ViT-S/16 backbone using PyTorch Hub.
def load_official_dino_vits16(pretrained: bool = True) -> nn.Module:

    """Load the official DINO ViT-S/16 architecture through ``torch.hub``."""

    # Try to download or load the DINO model from the local torch.hub cache.
    try:

        backbone = torch.hub.load(

            "facebookresearch/dino:main",

            "dino_vits16",

            pretrained=pretrained,

        )

    # Convert loading problems into a clearer project-specific error message.
    except Exception as exc:

        raise RuntimeError(

            "Failed to load the official DINO ViT-S/16 model from "

            "facebookresearch/dino. Check the internet connection and "

            "the local torch.hub cache."

        ) from exc

    # Verify that torch.hub returned a valid PyTorch model.
    if not isinstance(backbone, nn.Module):

        raise RuntimeError(

            "torch.hub did not return a valid PyTorch model."

        )

    # Return the loaded DINO backbone.
    return backbone


# Build the complete DINO ViT-S/16 model used for CIFAR-100 classification.
def build_dino_vits16_cifar100(

    num_classes: int = 100,

    freeze_backbone: bool = False,

    pretrained: bool = True,

) -> DinoViTS16CIFAR100:

    """Load DINO and return the CIFAR-100 classifier wrapper."""

    # Load the official DINO ViT-S/16 backbone.
    backbone = load_official_dino_vits16(pretrained=pretrained)

    # Attach the CIFAR-100 classification head and return the complete model.
    return DinoViTS16CIFAR100(

        backbone=backbone,

        num_classes=num_classes,

        freeze_backbone=freeze_backbone,

    )