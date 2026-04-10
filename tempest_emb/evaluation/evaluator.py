from typing import Tuple

import torch

from tempest_emb.data.negative_sampler import sample_negatives
from tempest_emb.models.embedding_store import EmbeddingStore
from tempest_emb.models.link_predictor import LinkPredictor
from tempest_emb.types import Batch


class Evaluator:
    """Streaming link-prediction evaluator.

    For each batch (called before ingestion by the Trainer):
      1. Sample negatives uniformly at random.
      2. Interleave positives with their negatives: [pos_1, neg_1_1..neg_1_K, pos_2, ...].
      3. Forward through EmbeddingStore + LinkPredictor under no_grad.
      4. Compute pessimistic MRR (ties counted against the positive).
    """

    def __init__(
        self,
        embedding_store: EmbeddingStore,
        link_predictor: LinkPredictor,
        num_nodes: int,
        negatives_per_positive: int,
        device: torch.device,
    ):
        self.embedding_store = embedding_store
        self.link_predictor = link_predictor
        self.num_nodes = num_nodes
        self.negatives_per_positive = negatives_per_positive
        self.device = device

    @torch.no_grad()
    def evaluate_batch(self, batch: Batch) -> Tuple[float, int]:
        """Rank each positive against its sampled negatives.

        Returns:
            (sum_of_reciprocal_ranks, num_positive_edges)
        """
        neg_src, neg_tgt = sample_negatives(
            batch, self.num_nodes, self.negatives_per_positive
        )

        pos_src = torch.from_numpy(batch.src).long().to(self.device)  # [B]
        pos_tgt = torch.from_numpy(batch.tgt).long().to(self.device)  # [B]
        neg_src_t = torch.from_numpy(neg_src).long().to(self.device)  # [B, K]
        neg_tgt_t = torch.from_numpy(neg_tgt).long().to(self.device)  # [B, K]

        B, K = neg_src_t.shape

        # Interleave: [B, 1+K] then flatten
        all_u = torch.cat([pos_src.unsqueeze(1), neg_src_t], dim=1).flatten()
        all_v = torch.cat([pos_tgt.unsqueeze(1), neg_tgt_t], dim=1).flatten()

        e_target_u = self.embedding_store.target(all_u)
        e_target_v = self.embedding_store.target(all_v)
        e_context_u = self.embedding_store.context(all_u)
        e_context_v = self.embedding_store.context(all_v)

        prob = self.link_predictor(
            e_target_u, e_target_v, e_context_u, e_context_v
        )  # [B * (1 + K)]

        scores = prob.view(B, 1 + K)
        pos_scores = scores[:, :1]          # [B, 1]
        neg_scores = scores[:, 1:]          # [B, K]

        # Pessimistic rank: ties counted against the positive.
        rank = (neg_scores >= pos_scores).sum(dim=1) + 1  # [B]
        reciprocal = 1.0 / rank.float()

        return reciprocal.sum().item(), B
