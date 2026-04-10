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
         scatter-based vectorised ranking for variable-K.
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

        all_u, all_v, counts, is_fixed_k = self._interleave(batch, neg_src, neg_tgt, B)

        # Single forward pass
        e_target_u = self.embedding_store.target(all_u)
        e_target_v = self.embedding_store.target(all_v)
        e_context_u = self.embedding_store.context(all_u)
        e_context_v = self.embedding_store.context(all_v)

        prob = self.link_predictor(
            e_target_u, e_target_v, e_context_u, e_context_v
        )

        # Pessimistic MRR
        total_rr = self._compute_mrr(prob, counts, B, is_fixed_k)
        return total_rr, B

    def _interleave(
        self,
        batch: Batch,
        neg_src,
        neg_tgt,
        B: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], bool]:
        """Build flat interleaved arrays and a counts vector.

        Layout: [pos_0, neg_0_1..neg_0_K0, pos_1, neg_1_1..neg_1_K1, ...]
        counts: [K0, K1, ...] — number of negatives per positive.

        Fixed-K ([B, K] arrays): vectorised cat + flatten.
        Variable-K (list-of-arrays): pre-allocated single-pass fill.
        """
        is_fixed_k = isinstance(neg_src, np.ndarray) and neg_src.ndim == 2

        if is_fixed_k:
            K = neg_src.shape[1]
            pos_src = torch.from_numpy(batch.src).long().to(self.device)
            pos_tgt = torch.from_numpy(batch.tgt).long().to(self.device)
            neg_src_t = torch.from_numpy(neg_src).long().to(self.device)
            neg_tgt_t = torch.from_numpy(neg_tgt).long().to(self.device)
            all_u = torch.cat([pos_src.unsqueeze(1), neg_src_t], dim=1).flatten()
            all_v = torch.cat([pos_tgt.unsqueeze(1), neg_tgt_t], dim=1).flatten()
            counts = [K] * B
        else:
            # Pre-allocate and fill in one pass (avoids 4*B tiny allocations)
            neg_lengths = [len(neg_tgt[i]) for i in range(B)]
            total = B + sum(neg_lengths)
            all_src_np = np.empty(total, dtype=np.int64)
            all_dst_np = np.empty(total, dtype=np.int64)
            counts = []
            offset = 0
            for i in range(B):
                K_i = neg_lengths[i]
                counts.append(K_i)
                all_src_np[offset] = batch.src[i]
                all_src_np[offset + 1 : offset + 1 + K_i] = neg_src[i]
                all_dst_np[offset] = batch.tgt[i]
                all_dst_np[offset + 1 : offset + 1 + K_i] = neg_tgt[i]
                offset += 1 + K_i
            all_u = torch.from_numpy(all_src_np).to(self.device)
            all_v = torch.from_numpy(all_dst_np).to(self.device)

        return all_u, all_v, counts, is_fixed_k

    def _compute_mrr(
        self,
        prob: torch.Tensor,
        counts: List[int],
        B: int,
        is_fixed_k: bool,
    ) -> float:
        """Pessimistic MRR from flat scores + counts vector.

        Fixed-K (all counts equal): reshape to [B, 1+K], vectorised sum.
        Variable-K: scatter-based vectorised ranking (no Python loop).
        """
        if is_fixed_k:
            K0 = counts[0]
            scores = prob.view(B, 1 + K0)
            pos_scores = scores[:, :1]      # [B, 1]
            neg_scores = scores[:, 1:]      # [B, K]
            rank = (neg_scores >= pos_scores).sum(dim=1) + 1  # [B]
            return (1.0 / rank.float()).sum().item()

        # Fully vectorised scatter-based ranking (avoids B GPU syncs)
        group_sizes = [1 + c for c in counts]
        group_sizes_t = torch.tensor(group_sizes, dtype=torch.long,
                                     device=prob.device)
        group_idx = torch.repeat_interleave(
            torch.arange(B, device=prob.device), group_sizes_t
        )

        # Positive score = first element of each group
        group_starts = torch.zeros(B, dtype=torch.long, device=prob.device)
        group_starts[1:] = group_sizes_t[:-1].cumsum(0)
        pos_scores = prob[group_starts]                        # [B]

        # Per-element comparison against its group's positive score;
        # scatter-sum gives rank = 1 (pos itself) + count(neg >= pos)
        beats_pos = (prob >= pos_scores[group_idx]).float()
        rank = torch.zeros(B, dtype=torch.float32, device=prob.device)
        rank.scatter_add_(0, group_idx, beats_pos)
        return (1.0 / rank).sum().item()
