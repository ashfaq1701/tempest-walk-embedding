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

        Args:
            batch:   Batch with positive (src, tgt) edges.
            neg_src: [B, num_neg] negative sources.
            neg_tgt: [B, num_neg] negative targets.

        Returns:
            Scalar loss value for logging.
        """
        pos_src = torch.from_numpy(batch.src).long().to(self.device)
        pos_tgt = torch.from_numpy(batch.tgt).long().to(self.device)
        neg_src_t = torch.from_numpy(neg_src).long().to(self.device).flatten()
        neg_tgt_t = torch.from_numpy(neg_tgt).long().to(self.device).flatten()

        all_u = torch.cat([pos_src, neg_src_t])
        all_v = torch.cat([pos_tgt, neg_tgt_t])

        # Frozen embedding lookups — no gradients flow to the tables
        with torch.no_grad():
            e_target_u = self.embedding_store.target(all_u)
            e_target_v = self.embedding_store.target(all_v)
            e_context_u = self.embedding_store.context(all_u)
            e_context_v = self.embedding_store.context(all_v)

        prob = self.link_predictor(e_target_u, e_target_v, e_context_u, e_context_v)

        labels = torch.cat([
            torch.ones(len(pos_src), device=self.device),
            torch.zeros(len(neg_src_t), device=self.device),
        ])

        loss = link_pred_loss(prob, labels)

        self.optimizer_link.zero_grad()
        loss.backward()
        self.optimizer_link.step()

        return loss.item()
