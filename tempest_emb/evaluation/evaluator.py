from typing import List, Optional, Tuple

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
      2. Pad to max-K, build [B, 1+max_K] rectangular batch.
      3. Single forward pass through EmbeddingStore + LinkPredictor.
      4. Masked pessimistic MRR — fully vectorised for both fixed-K
         and variable-K negatives.
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

        neg_src_padded, neg_tgt_padded, mask = self._pad_negatives(
            neg_src, neg_tgt
        )

        B, max_K = neg_src_padded.shape

        # Build [B, 1+max_K] interleaved layout, then flatten
        pos_src = torch.from_numpy(batch.src).long().to(self.device)
        pos_tgt = torch.from_numpy(batch.tgt).long().to(self.device)
        neg_src_t = torch.from_numpy(neg_src_padded).long().to(self.device)
        neg_tgt_t = torch.from_numpy(neg_tgt_padded).long().to(self.device)

        all_u = torch.cat([pos_src.unsqueeze(1), neg_src_t], dim=1).flatten()
        all_v = torch.cat([pos_tgt.unsqueeze(1), neg_tgt_t], dim=1).flatten()

        # Single forward pass
        e_target_u = self.embedding_store.target(all_u)
        e_target_v = self.embedding_store.target(all_v)
        e_context_u = self.embedding_store.context(all_u)
        e_context_v = self.embedding_store.context(all_v)

        prob = self.link_predictor(
            e_target_u, e_target_v, e_context_u, e_context_v
        )  # [B * (1 + max_K)]

        scores = prob.view(B, 1 + max_K)
        pos_scores = scores[:, :1]      # [B, 1]
        neg_scores = scores[:, 1:]      # [B, max_K]

        # Pessimistic rank: count negatives >= positive, ignoring padding
        if mask is not None:
            mask_t = torch.from_numpy(mask).to(self.device)  # [B, max_K]
            rank = ((neg_scores >= pos_scores) & mask_t).sum(dim=1) + 1
        else:
            rank = (neg_scores >= pos_scores).sum(dim=1) + 1  # [B]

        total_rr = (1.0 / rank.float()).sum().item()
        return total_rr, B

    @staticmethod
    def _pad_negatives(
        neg_src, neg_tgt
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Normalise negatives to [B, max_K] arrays with optional mask.

        Fixed-K ([B, K] arrays): returned as-is, mask is None.
        Variable-K (list-of-arrays): padded to max_K with zeros,
        mask[i, j] = j < K_i.
        """
        if isinstance(neg_src, np.ndarray) and neg_src.ndim == 2:
            return neg_src, neg_tgt, None

        B = len(neg_src)
        lengths = [len(neg_tgt[i]) for i in range(B)]
        max_K = max(lengths)

        src_padded = np.zeros((B, max_K), dtype=np.int32)
        tgt_padded = np.zeros((B, max_K), dtype=np.int32)
        mask = np.zeros((B, max_K), dtype=bool)

        for i in range(B):
            K_i = lengths[i]
            src_padded[i, :K_i] = neg_src[i]
            tgt_padded[i, :K_i] = neg_tgt[i]
            mask[i, :K_i] = True

        return src_padded, tgt_padded, mask
