from typing import Tuple

import numpy as np

from tempest_emb.types import Batch


def sample_negatives(
    batch: Batch, num_nodes: int, num_neg_per_pos: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample random negative edges for a batch of positive edges.

    For each positive edge, the source is kept and the target is replaced
    with a uniformly random node.  Collision with actual positives is
    negligible for any reasonably sized graph.

    Returns:
        (neg_src, neg_tgt) each shaped [B, num_neg_per_pos] int32.
    """
    B = len(batch.src)
    neg_src = np.broadcast_to(
        batch.src[:, None], (B, num_neg_per_pos)
    ).astype(np.int32, copy=True)
    neg_tgt = np.random.randint(
        0, num_nodes, (B, num_neg_per_pos), dtype=np.int32
    )
    return neg_src, neg_tgt
