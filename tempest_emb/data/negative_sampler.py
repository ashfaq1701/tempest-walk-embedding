import abc
import pickle
from typing import List, Tuple

import numpy as np

from tempest_emb.types import Batch


class NegativeSampler(abc.ABC):
    """Base class for negative samplers.

    All samplers expose a single method: sample(batch) → (neg_src, neg_tgt).
    """

    @abc.abstractmethod
    def sample(
        self, batch: Batch
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample negatives for each positive edge in the batch.

        Returns:
            (neg_src, neg_tgt) — each shaped [B, K] (int32) for fixed-K
            samplers, or list-of-ndarrays for variable-K samplers.
        """
        ...


class UniformNegativeSampler(NegativeSampler):
    """Uniform random negative sampler (the original default).

    For each positive edge, keeps the source and replaces the target
    with a uniformly random node.  Collision with actual positives is
    negligible for any reasonably sized graph.
    """

    def __init__(self, num_nodes: int, num_neg_per_pos: int):
        self.num_nodes = num_nodes
        self.num_neg_per_pos = num_neg_per_pos

    def sample(
        self, batch: Batch
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (neg_src, neg_tgt) each shaped [B, num_neg_per_pos] int32."""
        B = len(batch.src)
        neg_src = np.broadcast_to(
            batch.src[:, None], (B, self.num_neg_per_pos)
        ).astype(np.int32, copy=True)
        neg_tgt = np.random.randint(
            0, self.num_nodes, (B, self.num_neg_per_pos), dtype=np.int32
        )
        return neg_src, neg_tgt


class FileNegativeSampler(NegativeSampler):
    """Negative sampler backed by a TGB-format pickle file.

    The pickle contains a dict keyed by (pos_src, pos_dst, pos_ts)
    with values that are 1-D numpy arrays of negative destination node IDs.
    The source for each negative is implicitly the positive's source.

    Variable N per positive is supported — returns list-of-ndarrays.
    """

    def __init__(self, pickle_path: str):
        self.pickle_path = pickle_path
        with open(pickle_path, "rb") as f:
            raw = pickle.load(f)
        # Normalise keys to plain Python ints for robust lookup
        self.eval_set = {
            (int(s), int(d), int(t)): np.asarray(negs)
            for (s, d, t), negs in raw.items()
        }

    def sample(
        self, batch: Batch
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Returns (neg_src_list, neg_tgt_list) — list-of-ndarrays, one per positive.

        neg_tgt_list[i] = array of negative destinations from the pickle.
        neg_src_list[i] = pos_src[i] repeated to match length.
        """
        neg_src_list: List[np.ndarray] = []
        neg_tgt_list: List[np.ndarray] = []
        for s, d, t in zip(batch.src, batch.tgt, batch.ts):
            key = (int(s), int(d), int(t))
            if key not in self.eval_set:
                raise KeyError(
                    f"Positive edge {key} not found in negative pickle "
                    f"'{self.pickle_path}'. This indicates a split mismatch — "
                    f"the pickle was generated with different split boundaries."
                )
            neg_dsts = self.eval_set[key]
            neg_src_list.append(np.full(len(neg_dsts), s, dtype=np.int32))
            neg_tgt_list.append(neg_dsts.astype(np.int32))
        return neg_src_list, neg_tgt_list
