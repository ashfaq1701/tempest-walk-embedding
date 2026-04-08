from typing import NamedTuple, Optional

import numpy as np
import torch


class Batch(NamedTuple):
    src: np.ndarray        # [B] int64 — source node IDs
    tgt: np.ndarray        # [B] int64 — target node IDs
    ts: np.ndarray         # [B] int64 — timestamps
    edge_feat: Optional[np.ndarray]  # [B, d_edge] float32 or None
    t_max: int             # max timestamp in this batch


class WalkData(NamedTuple):
    nodes: torch.Tensor       # [W, L] int32, padding=-1
    timestamps: torch.Tensor  # [W, L] int64, last=INT_MAX sentinel
    lens: torch.Tensor        # [W] int64 — valid walk lengths
    edge_feats: Optional[torch.Tensor]  # [W, L-1, d_edge] float32 or None


class SplitData(NamedTuple):
    sources: np.ndarray       # [E] int64
    destinations: np.ndarray  # [E] int64
    timestamps: np.ndarray    # [E] int64
    edge_feat: Optional[np.ndarray]  # [E, d_edge] float32 or None
