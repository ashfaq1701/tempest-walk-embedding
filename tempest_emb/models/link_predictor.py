import torch
import torch.nn as nn


class LinkPredictor(nn.Module):
    """MLP for link prediction: [E_target[u] | E_target[v] | E_context[u] | E_context[v]] → probability."""

    def __init__(self, d_emb: int, d_hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 * d_emb, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, e_target_u: torch.Tensor, e_target_v: torch.Tensor,
                e_context_u: torch.Tensor, e_context_v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e_target_u:  [B, d_emb]
            e_target_v:  [B, d_emb]
            e_context_u: [B, d_emb]
            e_context_v: [B, d_emb]

        Returns:
            probabilities: [B]
        """
        x = torch.cat([e_target_u, e_target_v, e_context_u, e_context_v], dim=-1)
        return torch.sigmoid(self.mlp(x)).squeeze(-1)
