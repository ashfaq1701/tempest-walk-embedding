from typing import Tuple

import numpy as np

from temporal_negative_edge_sampler import NegativeEdgeSampler

from tempest_emb.config import Config
from tempest_emb.types import Batch


class NegativeSampler:
    """Wrapper around NegativeEdgeSampler for link prediction only.

    Not used for embedding training. Accumulates history across all batches.
    """

    def __init__(self, config: Config, is_directed: bool = True):
        self.sampler = NegativeEdgeSampler(
            is_directed=is_directed,
            num_neg_per_pos=config.link_pred_negatives_per_positive,
        )
        self.num_neg_per_pos = config.link_pred_negatives_per_positive

    def add_batch(self, batch: Batch) -> None:
        self.sampler.add_batch(batch.src, batch.tgt, batch.ts)

    def sample(self, batch: Batch) -> Tuple[np.ndarray, np.ndarray]:
        """Sample negatives for a batch of positive edges.

        Returns:
            (neg_src, neg_tgt) each shaped [B, num_neg_per_pos] int32.
        """
        neg_dict = self.sampler.sample_negatives()
        batch_size = len(batch.src)
        neg_src = neg_dict["sources"].reshape(batch_size, self.num_neg_per_pos)
        neg_tgt = neg_dict["targets"].reshape(batch_size, self.num_neg_per_pos)
        return neg_src, neg_tgt
