from typing import Tuple

import torch

from tempest_emb.data.negative_sampler import NegativeSampler
from tempest_emb.models.embedding_store import EmbeddingStore
from tempest_emb.models.link_predictor import LinkPredictor
from tempest_emb.types import Batch


class Evaluator:
    """Streaming link-prediction evaluator.

    For each batch (called before ingestion by the Trainer):
      1. Sample `eval_negatives_per_positive` negatives from the shared
         eval sampler (which carries full stream history).
      2. Interleave positives with their own negatives so each positive is
         grouped with its K negatives: [pos_1, neg_1_1..neg_1_K, pos_2, ...].
      3. Forward a single batch through EmbeddingStore + LinkPredictor
         under `torch.no_grad()`.
      4. Reshape scores to [B, 1+K], compute the pessimistic rank of each
         positive (ties counted against the positive), and aggregate
         sum of reciprocal ranks.

    The Trainer calls `evaluate_batch(batch)` first, then ingests the batch
    (updating both the walk graph and the eval sampler for subsequent calls).
    """

    def __init__(
        self,
        embedding_store: EmbeddingStore,
        link_predictor: LinkPredictor,
        neg_sampler_eval: NegativeSampler,
        device: torch.device,
    ):
        self.embedding_store = embedding_store
        self.link_predictor = link_predictor
        self.neg_sampler_eval = neg_sampler_eval
        self.device = device

    @torch.no_grad()
    def evaluate_batch(self, batch: Batch) -> Tuple[float, int]:
        """Rank each positive in the batch against its sampled negatives.

        Returns:
            (sum_of_reciprocal_ranks, num_positive_edges)
            The caller aggregates across batches and divides at the end.
        """
        neg_src, neg_tgt = self.neg_sampler_eval.sample(batch)  # [B, K]

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
