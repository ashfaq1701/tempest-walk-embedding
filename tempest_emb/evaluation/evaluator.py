from typing import List, Tuple, Union

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
      2. Flatten all (positive, negatives) into a single forward pass.
      3. Compute pessimistic MRR.

    Fixed-K path (uniform negatives): fully vectorised construction and
    ranking via [B, 1+K] reshape.  Variable-K path (TGB pickle): builds
    flat arrays in a loop, single forward pass, then per-group ranking.
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
        fixed_k = isinstance(neg_src, np.ndarray) and neg_src.ndim == 2

        # --- build flat src/dst tensors + forward pass ---
        if fixed_k:
            all_u, all_v, K = self._build_fixed_k(batch, neg_src, neg_tgt)
        else:
            all_u, all_v, group_sizes = self._build_variable_k(
                batch, neg_src, neg_tgt, B
            )

        e_target_u = self.embedding_store.target(all_u)
        e_target_v = self.embedding_store.target(all_v)
        e_context_u = self.embedding_store.context(all_u)
        e_context_v = self.embedding_store.context(all_v)

        prob = self.link_predictor(
            e_target_u, e_target_v, e_context_u, e_context_v
        )

        # --- compute pessimistic MRR ---
        if fixed_k:
            scores = prob.view(B, 1 + K)
            pos_scores = scores[:, :1]      # [B, 1]
            neg_scores = scores[:, 1:]      # [B, K]
            rank = (neg_scores >= pos_scores).sum(dim=1) + 1  # [B]
            total_rr = (1.0 / rank.float()).sum().item()
        else:
            total_rr = 0.0
            offset = 0
            for size in group_sizes:
                group = prob[offset : offset + size]
                rank = (group[1:] >= group[0]).sum() + 1
                total_rr += (1.0 / rank.float()).item()
                offset += size

        return total_rr, B

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    def _build_fixed_k(
        self,
        batch: Batch,
        neg_src: np.ndarray,
        neg_tgt: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Vectorised [B, 1+K] interleave → flat tensors. Returns (all_u, all_v, K)."""
        pos_src = torch.from_numpy(batch.src).long().to(self.device)
        pos_tgt = torch.from_numpy(batch.tgt).long().to(self.device)
        neg_src_t = torch.from_numpy(neg_src).long().to(self.device)
        neg_tgt_t = torch.from_numpy(neg_tgt).long().to(self.device)
        K = neg_src_t.shape[1]
        all_u = torch.cat([pos_src.unsqueeze(1), neg_src_t], dim=1).flatten()
        all_v = torch.cat([pos_tgt.unsqueeze(1), neg_tgt_t], dim=1).flatten()
        return all_u, all_v, K

    def _build_variable_k(
        self,
        batch: Batch,
        neg_src_list: List[np.ndarray],
        neg_tgt_list: List[np.ndarray],
        B: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Per-positive concat → flat tensors. Returns (all_u, all_v, group_sizes)."""
        group_sizes: List[int] = []
        src_parts: List[np.ndarray] = []
        dst_parts: List[np.ndarray] = []
        for i in range(B):
            K_i = len(neg_tgt_list[i])
            group_sizes.append(1 + K_i)
            src_parts.append(np.array([batch.src[i]], dtype=np.int32))
            src_parts.append(np.asarray(neg_src_list[i], dtype=np.int32))
            dst_parts.append(np.array([batch.tgt[i]], dtype=np.int32))
            dst_parts.append(np.asarray(neg_tgt_list[i], dtype=np.int32))
        all_u = torch.from_numpy(np.concatenate(src_parts)).long().to(self.device)
        all_v = torch.from_numpy(np.concatenate(dst_parts)).long().to(self.device)
        return all_u, all_v, group_sizes
