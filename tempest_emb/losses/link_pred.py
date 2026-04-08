import torch
import torch.nn.functional as F


def link_pred_loss(prob: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy loss for link prediction.

    Args:
        prob:   [B] predicted probabilities (after sigmoid).
        labels: [B] binary labels (1=positive, 0=negative).

    Returns:
        Scalar BCE loss.
    """
    return F.binary_cross_entropy(prob, labels)
