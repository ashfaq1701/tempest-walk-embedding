from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd

from tempest_emb.config import Config
from tempest_emb.types import Batch, SplitData


def load_dataset(
    config: Config,
) -> Tuple[SplitData, SplitData, SplitData, Optional[np.ndarray]]:
    """Load dataset and return chronological train/val/test splits + node features.

    Expects files in config.data_dir:
        {dataset}.csv          — columns: u, i, ts, idx
        {dataset}_edges.npy    — edge features indexed by idx
        {dataset}_node.npy     — node features indexed by dense node ID

    Returns:
        (train_data, val_data, test_data, node_feat)
        node_feat is [N, d_node] float32 or None.
    """
    data_dir = Path(config.data_dir)
    name = config.dataset

    # Load CSV
    csv_path = data_dir / f"{name}.csv"
    df = pd.read_csv(csv_path)
    sources = df["u"].values.astype(np.int64)
    destinations = df["i"].values.astype(np.int64)
    timestamps = df["ts"].values.astype(np.int64)
    idx = df["idx"].values.astype(np.int64)

    assert np.all(np.diff(timestamps) >= 0), "CSV must be sorted by timestamp"

    # Load edge features
    edge_feat_path = data_dir / f"{name}_edges.npy"
    if edge_feat_path.exists():
        all_edge_feat = np.load(edge_feat_path).astype(np.float32)
        edge_feat = all_edge_feat[idx]
    else:
        edge_feat = None

    # Load node features
    node_feat_path = data_dir / f"{name}_node.npy"
    if node_feat_path.exists():
        node_feat = np.load(node_feat_path).astype(np.float32)
    else:
        node_feat = None

    # Chronological split (TGB-identical np.quantile)
    train_mask, val_mask, test_mask = chronological_split(
        timestamps, train_ratio=config.train_ratio, val_ratio=config.val_ratio
    )

    def _apply_mask(mask: np.ndarray) -> SplitData:
        ef = edge_feat[mask] if edge_feat is not None else None
        return SplitData(
            sources=sources[mask],
            destinations=destinations[mask],
            timestamps=timestamps[mask],
            edge_feat=ef,
        )

    return _apply_mask(train_mask), _apply_mask(val_mask), _apply_mask(test_mask), node_feat


def chronological_split(
    timestamps: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TGB-identical temporal quantile split.

    Uses np.quantile on the per-edge timestamp array, producing split
    boundaries that are byte-for-byte identical to both TGB
    (tgb/linkproppred/dataset.py:400-428) and Neural Temporal Walks
    (main.py:33,39-41) for matching ratios.

    Returns:
        (train_mask, val_mask, test_mask) — boolean arrays.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    val_time, test_time = np.quantile(
        timestamps, [1.0 - val_ratio - test_ratio, 1.0 - test_ratio]
    )

    train_mask = timestamps <= val_time
    val_mask = (timestamps <= test_time) & (timestamps > val_time)
    test_mask = timestamps > test_time

    n = len(timestamps)
    print(
        f"Split: train={train_mask.sum()} ({train_mask.sum()/n:.1%}), "
        f"val={val_mask.sum()} ({val_mask.sum()/n:.1%}), "
        f"test={test_mask.sum()} ({test_mask.sum()/n:.1%})"
    )

    return train_mask, val_mask, test_mask


def create_batches(
    split_data: SplitData,
    target_batch_size: int,
) -> Iterator[Batch]:
    """Yield timestamp-respecting batches from a split.

    Rules:
        1. All edges sharing a timestamp stay in the same batch.
        2. Accumulate chronologically until the next group would exceed target_batch_size.
        3. A single timestamp exceeding target_batch_size becomes its own oversized batch.
    """
    sources = split_data.sources
    destinations = split_data.destinations
    timestamps = split_data.timestamps
    edge_feat = split_data.edge_feat
    n = len(timestamps)

    if n == 0:
        return

    # Find timestamp group boundaries
    ts_change = np.where(np.diff(timestamps) != 0)[0] + 1
    group_starts = np.concatenate([[0], ts_change])
    group_ends = np.concatenate([ts_change, [n]])

    batch_start_group = 0

    for i in range(len(group_starts)):
        edges_so_far = group_ends[i] - group_starts[batch_start_group]

        if edges_so_far > target_batch_size and i > batch_start_group:
            yield _make_batch(
                sources, destinations, timestamps, edge_feat,
                start=group_starts[batch_start_group],
                end=group_starts[i],
            )
            batch_start_group = i

    # Emit remaining
    if group_starts[batch_start_group] < n:
        yield _make_batch(
            sources, destinations, timestamps, edge_feat,
            start=group_starts[batch_start_group],
            end=n,
        )


def _make_batch(
    sources: np.ndarray,
    destinations: np.ndarray,
    timestamps: np.ndarray,
    edge_feat: Optional[np.ndarray],
    start: int,
    end: int,
) -> Batch:
    ef = edge_feat[start:end] if edge_feat is not None else None
    return Batch(
        src=sources[start:end],
        tgt=destinations[start:end],
        ts=timestamps[start:end],
        edge_feat=ef,
        t_max=int(timestamps[end - 1]),
    )
