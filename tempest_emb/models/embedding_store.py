from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class EmbeddingStore(nn.Module):
    """Two embedding tables with dynamic node initialization.

    - E_target: "what is this node?" — identity representation
    - E_context: "what context does this node provide?" — contextual role

    Pre-allocates `num_nodes` slots. Every slot starts with a small random value
    and is marked uninitialized. When a node is first seen in the stream,
    `initialize_nodes()` is called to overwrite its slot with either:
      - W_init projection of node features (if features exist), or
      - Detached copy of a first-neighbor's embedding (if no features).

    If node features exist, they are stored as a buffer so the alignment loss
    can concatenate them with embeddings during training.
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
        nn.init.normal_(self.E_target.weight, std=0.01)
        nn.init.normal_(self.E_context.weight, std=0.01)

        # Tracks whether each node has been properly initialized.
        self.register_buffer(
            "initialized", torch.zeros(num_nodes, dtype=torch.bool)
        )

        if node_feat is not None:
            self.d_node = int(node_feat.shape[1])
            self.register_buffer(
                "node_feat", torch.from_numpy(node_feat).float()
            )
            self.W_init = nn.Linear(self.d_node, d_emb, bias=False)
            self.W_init_ctx = nn.Linear(self.d_node, d_emb, bias=False)
        else:
            self.d_node = 0
            self.node_feat = None
            self.W_init = None
            self.W_init_ctx = None

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

    def find_uninitialized(self, ids: torch.Tensor) -> torch.Tensor:
        """Return the subset of `ids` that have not been initialized yet."""
        ids = ids.long().to(self.initialized.device)
        unique_ids = ids.unique()
        mask = ~self.initialized[unique_ids]
        return unique_ids[mask]

    def initialize_nodes(
        self,
        new_ids: torch.Tensor,
        neighbor_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize fresh nodes' embeddings.

        Args:
            new_ids:      [K] IDs of nodes being seen for the first time.
            neighbor_ids: [K] IDs of a reference neighbor per new node.
                          Required only when node features are unavailable.

        When node features exist: `E_target[new] = W_init(x_node[new])`,
        `E_context[new] = W_init_ctx(x_node[new])`.

        When node features are absent: `E_target[new] = E_target[neighbor].detach()`,
        `E_context[new] = E_context[neighbor].detach()`.
        """
        if new_ids.numel() == 0:
            return
        device = self.E_target.weight.device
        new_ids = new_ids.long().to(device)

        with torch.no_grad():
            if self.has_node_features:
                feats = self.node_feat[new_ids]
                self.E_target.weight.data[new_ids] = self.W_init(feats)
                self.E_context.weight.data[new_ids] = self.W_init_ctx(feats)
            else:
                assert neighbor_ids is not None, (
                    "neighbor_ids required when no node features are present"
                )
                neighbor_ids = neighbor_ids.long().to(device)
                self.E_target.weight.data[new_ids] = (
                    self.E_target.weight.data[neighbor_ids].clone()
                )
                self.E_context.weight.data[new_ids] = (
                    self.E_context.weight.data[neighbor_ids].clone()
                )

        self.initialized[new_ids] = True
