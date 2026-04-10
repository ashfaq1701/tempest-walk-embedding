import torch

from temporal_random_walk import TemporalRandomWalk

from tempest_emb.config import Config
from tempest_emb.types import WalkData


class WalkGenerator:
    """Wrapper around Tempest's TemporalRandomWalk for backward walk generation."""

    def __init__(self, config: Config):
        self.trw = TemporalRandomWalk(is_directed=config.is_directed, use_gpu=config.use_gpu, enable_weight_computation=True, timescale_bound=300)
        self.max_walk_len = config.max_walk_len
        self.num_walks_per_node = config.num_walks_per_node
        self.walk_bias = config.walk_bias

    def add_edges(self, sources, targets, timestamps) -> None:
        self.trw.add_multiple_edges(sources, targets, timestamps)

    def generate(self) -> WalkData:
        """Generate backward walks for the last ingested batch.

        Returns:
            WalkData with nodes, timestamps, lens, edge_feats as tensors.
        """
        nodes, ts, lens, ef = self.trw.get_random_walks_and_times_for_last_batch(
            max_walk_len=self.max_walk_len,
            walk_bias=self.walk_bias,
            num_walks_per_node=self.num_walks_per_node,
            walk_direction="Backward_In_Time",
        )

        return WalkData(
            nodes=torch.from_numpy(nodes),            # [W, L] int32
            timestamps=torch.from_numpy(ts),           # [W, L] int64
            lens=torch.from_numpy(lens).to(torch.int64),  # uint64 → int64
            edge_feats=torch.from_numpy(ef) if ef is not None else None,
        )
