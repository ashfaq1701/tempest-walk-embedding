from typing import Tuple

import torch

from tempest_emb.config import Config
from tempest_emb.losses.alignment import alignment_loss
from tempest_emb.losses.uniformity import uniformity_loss
from tempest_emb.models.embedding_store import EmbeddingStore
from tempest_emb.types import WalkData


class EmbeddingTrainer:
    """Trains E_target and E_context via alignment + uniformity on backward walks.

    Step:
      1. Look up E_target at position 0, E_context at positions 1..L-1.
      2. If node features exist, also fetch them for concatenation in alignment.
      3. Compute L_align (alignment) and L_uniform (uniformity) losses.
      4. L_total = L_align + eta * L_uniform.
      5. Backward through L_total, step optimizer_emb.
    """

    def __init__(
        self,
        embedding_store: EmbeddingStore,
        optimizer_emb: torch.optim.Optimizer,
        config: Config,
        device: torch.device,
    ):
        self.embedding_store = embedding_store
        self.optimizer_emb = optimizer_emb
        self.device = device
        self.beta = config.temporal_decay_exp
        self.eta_uniform = config.eta_uniform
        self.uniformity_temperature = config.uniformity_temperature
        self.uniformity_cap = config.uniformity_cap

    def step(self, walks: WalkData, t_max: int) -> Tuple[float, float, float]:
        """Run one training step on the given walks.

        Args:
            walks: WalkData produced by WalkGenerator.generate().
            t_max: Max timestamp in the current batch (used by alignment weights).

        Returns:
            (l_align, l_uniform, l_total) scalar loss values for logging.
        """
        nodes = walks.nodes.long().to(self.device)              # [W, L]
        timestamps = walks.timestamps.to(self.device)            # [W, L]
        lens = walks.lens.to(self.device)                        # [W]

        target_ids = nodes[:, 0]                                 # [W]
        context_ids = nodes[:, 1:]                               # [W, L-1]

        # Padding cells hold -1. Clamp to 0 for safe lookups; the alignment
        # mask zeros out their contribution to the loss.
        context_ids_safe = context_ids.clamp(min=0)

        e_target = self.embedding_store.target(target_ids)       # [W, d_emb]
        e_context = self.embedding_store.context(context_ids_safe)  # [W, L-1, d_emb]

        if self.embedding_store.has_node_features:
            nf_target = self.embedding_store.get_node_feat(target_ids)        # [W, d_node]
            nf_context = self.embedding_store.get_node_feat(context_ids_safe) # [W, L-1, d_node]
        else:
            nf_target = None
            nf_context = None

        l_align = alignment_loss(
            e_target=e_target,
            e_context=e_context,
            timestamps=timestamps,
            lens=lens,
            t_max=t_max,
            beta=self.beta,
            nf_target=nf_target,
            nf_context=nf_context,
        )

        l_uniform = uniformity_loss(
            walk_nodes=nodes,
            e_target=self.embedding_store.E_target,
            temperature=self.uniformity_temperature,
            cap=self.uniformity_cap,
        )

        l_total = l_align + self.eta_uniform * l_uniform

        self.optimizer_emb.zero_grad()
        l_total.backward()
        self.optimizer_emb.step()

        return l_align.item(), l_uniform.item(), l_total.item()
