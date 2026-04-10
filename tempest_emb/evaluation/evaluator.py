from typing import Tuple

import numpy as np
import torch

from tempest_emb.data.negative_sampler import (
    FileNegativeSampler,
    NegativeSampler,
)
from tempest_emb.models.embedding_store import EmbeddingStore
from tempest_emb.models.link_predictor import LinkPredictor
from tempest_emb.types import Batch


class Evaluator:
    """Streaming link-prediction evaluator.

    For each batch (called before ingestion by the Trainer):
      1. Sample negatives via the injected NegativeSampler.
      2. Score positives against their negatives.
      3. Compute pessimistic MRR (ties counted against the positive).

    Supports both fixed-K negatives (UniformNegativeSampler, [B, K] arrays)
    and variable-K negatives (FileNegativeSampler, list-of-ndarrays).
    """

    def __init__(
        self,
        embedding_store: EmbeddingStore,
        link_predictor: LinkPredictor,
        neg_sampler: NegativeSampler,
        device: torch.device,
    ):
        self.embedding_store = embedding_store
        self.link_predictor = link_predictor
        self.neg_sampler = neg_sampler
        self.device = device

    @torch.no_grad()
    def evaluate_batch(self, batch: Batch) -> Tuple[float, int]:
        """Rank each positive against its sampled negatives.

        Returns:
            (sum_of_reciprocal_ranks, num_positive_edges)
        """
        neg_src, neg_tgt = self.neg_sampler.sample(batch)

        if isinstance(self.neg_sampler, FileNegativeSampler):
            return self._evaluate_variable_k(batch, neg_src, neg_tgt)
        else:
            return self._evaluate_fixed_k(batch, neg_src, neg_tgt)

    def _evaluate_fixed_k(
        self,
        batch: Batch,
        neg_src: np.ndarray,
        neg_tgt: np.ndarray,
    ) -> Tuple[float, int]:
        """Vectorised evaluation when all positives have the same K negatives."""
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

    def _evaluate_variable_k(
        self,
        batch: Batch,
        neg_src_list,
        neg_tgt_list,
    ) -> Tuple[float, int]:
        """Per-positive evaluation when K varies (TGB pickle negatives)."""
        total_rr = 0.0
        B = len(batch.src)

        for i in range(B):
            pos_s = int(batch.src[i])
            pos_d = int(batch.tgt[i])
            neg_dsts = neg_tgt_list[i]
            K_i = len(neg_dsts)

            # Build [1 + K_i] source and destination arrays
            src_arr = np.concatenate(
                [np.array([pos_s], dtype=np.int32), neg_src_list[i]]
            )
            dst_arr = np.concatenate(
                [np.array([pos_d], dtype=np.int32), neg_dsts]
            )

            all_u = torch.from_numpy(src_arr).long().to(self.device)
            all_v = torch.from_numpy(dst_arr).long().to(self.device)

            e_target_u = self.embedding_store.target(all_u)
            e_target_v = self.embedding_store.target(all_v)
            e_context_u = self.embedding_store.context(all_u)
            e_context_v = self.embedding_store.context(all_v)

            prob = self.link_predictor(
                e_target_u, e_target_v, e_context_u, e_context_v
            )  # [1 + K_i]

            pos_score = prob[0]
            neg_scores = prob[1:]

            rank = (neg_scores >= pos_score).sum() + 1
            total_rr += (1.0 / rank.float()).item()

        return total_rr, B
