from typing import Optional

import torch
import torch.nn.functional as F


def alignment_loss(
    e_target: torch.Tensor,
    e_context: torch.Tensor,
    timestamps: torch.Tensor,
    lens: torch.Tensor,
    t_max: int,
    beta: float,
    nf_target: Optional[torch.Tensor] = None,
    nf_context: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Alignment loss on the [W, L-1] target–context grid.

    Pulls walk-connected pairs (target at position 0, context at positions 1..L-1)
    together via weighted cosine similarity. Weights decay by position (1/k)
    and by temporal lag via (1 + Δt)^(-β).

    Operates on pre-looked-up embeddings. If raw node features are provided,
    they are concatenated along the last dimension before L2 normalization
    (primary usage of node features per the design).

    Args:
        e_target:   [W, d_emb]          E_target[n0] per walk.
        e_context:  [W, L-1, d_emb]     E_context[n1..n_{L-1}] per walk.
        timestamps: [W, L]              walk timestamps (last entry is sentinel).
        lens:       [W]                 valid walk length per walk.
        t_max:      scalar              max timestamp in the current batch.
        beta:       scalar              temporal decay exponent.
        nf_target:  [W, d_node] or None    raw node features at target.
        nf_context: [W, L-1, d_node] or None  raw node features at context.

    Returns:
        Scalar alignment loss.
    """
    # Step 1: Optional node feature concatenation
    if nf_target is not None and nf_context is not None:
        e_target = torch.cat([e_target, nf_target], dim=-1)
        e_context = torch.cat([e_context, nf_context], dim=-1)

    # Step 2: L2 normalize
    e_target = F.normalize(e_target, dim=-1)
    e_context = F.normalize(e_context, dim=-1)

    # Step 3: Batched cosine similarity — one bmm
    cos_sim = torch.bmm(
        e_target.unsqueeze(1),         # [W, 1, d]
        e_context.transpose(1, 2),     # [W, d, L-1]
    ).squeeze(1)                       # [W, L-1]

    # Step 4: Temporal weights
    W, L = timestamps.shape
    device = e_target.device

    delta_t = (t_max - timestamps[:, :-1]).float().clamp(min=0.0)      # [W, L-1]
    k = torch.arange(1, L, device=device, dtype=torch.float32)         # [L-1]
    w = (1.0 / k).unsqueeze(0) * (1.0 + delta_t).pow(-beta)            # [W, L-1]

    # Step 5: Mask invalid positions (context positions k with k >= len)
    positions = torch.arange(1, L, device=device)                      # [L-1]
    mask = positions.unsqueeze(0) < lens.unsqueeze(1).to(device)       # [W, L-1]
    w = w * mask.float()

    # Step 6: Weighted loss normalized by the number of valid pairs
    num_valid = mask.sum().clamp(min=1)
    return (w * (1.0 - cos_sim)).sum() / num_valid
