from typing import List, Tuple

import numpy as np
import torch

from tempest_emb.data.negative_sampler import NegativeSampler
from tempest_emb.models.embedding_store import EmbeddingStore
from tempest_emb.models.link_predictor import LinkPredictor
from tempest_emb.types import Batch


class Evaluator:
    """Streaming link-prediction evaluator.

    For each batch (called before ingestion by the Trainer):
      1. Sample negatives via the injected NegativeSampler.
      2. Flatten positives + negatives into one interleaved array with a
         counts vector: [pos_0, neg_0_1..neg_0_K0, pos_1, neg_1_1..neg_1_K1, ...]
      3. Single forward pass through EmbeddingStore + LinkPredictor.
      4. Pessimistic MRR — reshape + vectorised sum for fixed-K,
         torch.split for variable-K.
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
        B = len(batch.src)

        all_u, all_v, counts = self._interleave(batch, neg_src, neg_tgt, B)

        # Single forward pass
        e_target_u = self.embedding_store.target(all_u)
        e_target_v = self.embedding_store.target(all_v)
        e_context_u = self.embedding_store.context(all_u)
        e_context_v = self.embedding_store.context(all_v)

        prob = self.link_predictor(
            e_target_u, e_target_v, e_context_u, e_context_v
        )

        # Pessimistic MRR
        total_rr = self._compute_mrr(prob, counts, B)
        return total_rr, B

    def _interleave(
        self,
        batch: Batch,
        neg_src,
        neg_tgt,
        B: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Build flat interleaved arrays and a counts vector.

        Layout: [pos_0, neg_0_1..neg_0_K0, pos_1, neg_1_1..neg_1_K1, ...]
        counts: [K0, K1, ...] — number of negatives per positive.

        Fixed-K ([B, K] arrays): vectorised cat + flatten.
        Variable-K (list-of-arrays): per-positive concat.
        """
        fixed_k = isinstance(neg_src, np.ndarray) and neg_src.ndim == 2

        if fixed_k:
            K = neg_src.shape[1]
            pos_src = torch.from_numpy(batch.src).long().to(self.device)
            pos_tgt = torch.from_numpy(batch.tgt).long().to(self.device)
            neg_src_t = torch.from_numpy(neg_src).long().to(self.device)
            neg_tgt_t = torch.from_numpy(neg_tgt).long().to(self.device)
            all_u = torch.cat([pos_src.unsqueeze(1), neg_src_t], dim=1).flatten()
            all_v = torch.cat([pos_tgt.unsqueeze(1), neg_tgt_t], dim=1).flatten()
            counts = [K] * B
        else:
            src_parts: List[np.ndarray] = []
            dst_parts: List[np.ndarray] = []
            counts = []
            for i in range(B):
                K_i = len(neg_tgt[i])
                counts.append(K_i)
                src_parts.append(np.array([batch.src[i]], dtype=np.int32))
                src_parts.append(np.asarray(neg_src[i], dtype=np.int32))
                dst_parts.append(np.array([batch.tgt[i]], dtype=np.int32))
                dst_parts.append(np.asarray(neg_tgt[i], dtype=np.int32))
            all_u = torch.from_numpy(np.concatenate(src_parts)).long().to(self.device)
            all_v = torch.from_numpy(np.concatenate(dst_parts)).long().to(self.device)

        return all_u, all_v, counts

    def _compute_mrr(
        self,
        prob: torch.Tensor,
        counts: List[int],
        B: int,
    ) -> float:
        """Pessimistic MRR from flat scores + counts vector.

        Fixed-K (all counts equal): reshape to [B, 1+K], vectorised sum.
        Variable-K: torch.split by group sizes, per-group ranking.
        """
        K0 = counts[0]
        fixed_k = all(c == K0 for c in counts)

        if fixed_k:
            scores = prob.view(B, 1 + K0)
            pos_scores = scores[:, :1]      # [B, 1]
            neg_scores = scores[:, 1:]      # [B, K]
            rank = (neg_scores >= pos_scores).sum(dim=1) + 1  # [B]
            return (1.0 / rank.float()).sum().item()

        group_sizes = [1 + c for c in counts]
        groups = torch.split(prob, group_sizes)
        total_rr = 0.0
        for group in groups:
            rank = (group[1:] >= group[0]).sum() + 1
            total_rr += (1.0 / rank.float()).item()
        return total_rr
