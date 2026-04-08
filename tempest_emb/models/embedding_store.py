from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class EmbeddingStore(nn.Module):
    """Two embedding tables with random initialization.

    - E_target: "what is this node?" — identity representation
    - E_context: "what context does this node provide?" — contextual role

    Both tables are pre-allocated to `num_nodes` slots and initialized with
    Xavier-uniform random values, scaled so the L2 norm of each row is roughly
    1/sqrt(d_emb). New nodes simply use their pre-initialized random slot —
    SGD moves them into a useful position over training.

    If node features are provided, they are stored as a buffer so the alignment
    loss can concatenate them with embeddings during training. They are NOT
    used to initialize the embedding tables.
    """

    def __init__(
        self,
        num_nodes: int,
        d_emb: int,
        node_feat: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_emb = d_emb

        self.E_target = nn.Embedding(num_nodes, d_emb)
        self.E_context = nn.Embedding(num_nodes, d_emb)
        nn.init.xavier_uniform_(self.E_target.weight)
        nn.init.xavier_uniform_(self.E_context.weight)

        if node_feat is not None:
            self.d_node = int(node_feat.shape[1])
            self.register_buffer(
                "node_feat", torch.from_numpy(node_feat).float()
            )
        else:
            self.d_node = 0
            self.node_feat = None

    @property
    def has_node_features(self) -> bool:
        return self.node_feat is not None

    def target(self, ids: torch.Tensor) -> torch.Tensor:
        """[K] → [K, d_emb]"""
        return self.E_target(ids.long())

    def context(self, ids: torch.Tensor) -> torch.Tensor:
        """[K] → [K, d_emb]"""
        return self.E_context(ids.long())

    def get_node_feat(self, ids: torch.Tensor) -> Optional[torch.Tensor]:
        """[K] → [K, d_node] or None"""
        if self.node_feat is None:
            return None
        return self.node_feat[ids.long()]
