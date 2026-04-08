import numpy as np
import torch

from tempest_emb.losses.link_pred import link_pred_loss
from tempest_emb.models.embedding_store import EmbeddingStore
from tempest_emb.models.link_predictor import LinkPredictor
from tempest_emb.types import Batch


class LinkPredTrainer:
    """Trains the link prediction MLP with frozen embeddings.

    Only `LinkPredictor` parameters receive gradients. Embeddings are read
    under `torch.no_grad()` so nothing flows back to the embedding tables.
    """

    def __init__(
        self,
        embedding_store: EmbeddingStore,
        link_predictor: LinkPredictor,
        optimizer_link: torch.optim.Optimizer,
        device: torch.device,
    ):
        self.embedding_store = embedding_store
        self.link_predictor = link_predictor
        self.optimizer_link = optimizer_link
        self.device = device

    def step(
        self,
        batch: Batch,
        neg_src: np.ndarray,
        neg_tgt: np.ndarray,
    ) -> float:
        """Train on one batch of positives and sampled negatives.

        Edges are organized in an interleaved layout so each positive is
        followed by its own sampled negatives:

            [pos_1, neg_1_1, ..., neg_1_K, pos_2, neg_2_1, ..., neg_2_K, ...]

        This layout is consistent with the evaluation ranking protocol and
        is produced purely via tensor ops (one cat + flatten, no loops).

        Args:
            batch:   Batch with positive (src, tgt) edges.
            neg_src: [B, num_neg] negative sources.
            neg_tgt: [B, num_neg] negative targets.

        Returns:
            Scalar loss value for logging.
        """
        pos_src = torch.from_numpy(batch.src).long().to(self.device)  # [B]
        pos_tgt = torch.from_numpy(batch.tgt).long().to(self.device)  # [B]
        neg_src_t = torch.from_numpy(neg_src).long().to(self.device)  # [B, K]
        neg_tgt_t = torch.from_numpy(neg_tgt).long().to(self.device)  # [B, K]

        B, K = neg_src_t.shape

        # Interleave: [B, 1 + K] → flatten to [B * (1 + K)]
        all_u = torch.cat([pos_src.unsqueeze(1), neg_src_t], dim=1).flatten()
        all_v = torch.cat([pos_tgt.unsqueeze(1), neg_tgt_t], dim=1).flatten()

        # Frozen embedding lookups — no gradients flow to the tables
        with torch.no_grad():
            e_target_u = self.embedding_store.target(all_u)
            e_target_v = self.embedding_store.target(all_v)
            e_context_u = self.embedding_store.context(all_u)
            e_context_v = self.embedding_store.context(all_v)

        prob = self.link_predictor(e_target_u, e_target_v, e_context_u, e_context_v)

        # Labels in the same interleaved order: each row is [1, 0, 0, ..., 0]
        labels = torch.cat([
            torch.ones(B, 1, device=self.device),
            torch.zeros(B, K, device=self.device),
        ], dim=1).flatten()

        loss = link_pred_loss(prob, labels)

        self.optimizer_link.zero_grad()
        loss.backward()
        self.optimizer_link.step()

        return loss.item()
