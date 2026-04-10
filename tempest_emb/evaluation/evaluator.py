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
      2. Flatten all (positive, negatives) groups into one forward pass.
      3. Split scores by group sizes and compute pessimistic MRR.

    Works identically for fixed-K (all groups size 1+K) and variable-K
    (groups of size 1+K_i).
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

        # Normalise to list-of-arrays so the same code handles both shapes.
        # Fixed-K: [B, K] ndarray → list of B rows, each length K.
        # Variable-K: already list-of-ndarrays.
        neg_src_list = self._to_list(neg_src, B)
        neg_tgt_list = self._to_list(neg_tgt, B)

        # Build flat src/dst arrays: [pos_0, neg_0_1..neg_0_K0, pos_1, ...]
        group_sizes: List[int] = []
        src_parts: List[np.ndarray] = []
        dst_parts: List[np.ndarray] = []

        for i in range(B):
            ns = neg_src_list[i]
            nt = neg_tgt_list[i]
            K_i = len(nt)
            group_sizes.append(1 + K_i)
            src_parts.append(np.array([batch.src[i]], dtype=np.int32))
            src_parts.append(np.asarray(ns, dtype=np.int32))
            dst_parts.append(np.array([batch.tgt[i]], dtype=np.int32))
            dst_parts.append(np.asarray(nt, dtype=np.int32))

        all_u = torch.from_numpy(np.concatenate(src_parts)).long().to(self.device)
        all_v = torch.from_numpy(np.concatenate(dst_parts)).long().to(self.device)

        # Single forward pass
        e_target_u = self.embedding_store.target(all_u)
        e_target_v = self.embedding_store.target(all_v)
        e_context_u = self.embedding_store.context(all_u)
        e_context_v = self.embedding_store.context(all_v)

        prob = self.link_predictor(
            e_target_u, e_target_v, e_context_u, e_context_v
        )

        # Split scores by group and compute pessimistic rank per positive
        total_rr = 0.0
        offset = 0
        for size in group_sizes:
            group = prob[offset : offset + size]
            pos_score = group[0]
            neg_scores = group[1:]
            rank = (neg_scores >= pos_score).sum() + 1
            total_rr += (1.0 / rank.float()).item()
            offset += size

        return total_rr, B

    @staticmethod
    def _to_list(
        arr: Union[np.ndarray, List[np.ndarray]], B: int
    ) -> List[np.ndarray]:
        """Normalise [B, K] array or list-of-arrays to list-of-arrays."""
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            return [arr[i] for i in range(B)]
        return arr
