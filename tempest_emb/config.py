from typing import Optional

from pydantic import BaseModel, field_validator


class Config(BaseModel):
    # Embeddings
    d_emb: int = 128

    # Link prediction MLP
    d_hidden_link: int = 128

    # Walks
    max_walk_len: int = 10
    num_walks_per_node: int = 5
    walk_bias: str = "Exponential"

    # Alignment loss
    temporal_decay_exp: float = 0.5  # β in (1+Δt)^(-β)

    # Uniformity loss
    eta_uniform: float = 1.0  # η coefficient
    uniformity_temperature: float = 2.0
    uniformity_cap: int = 20000

    # Link prediction loss
    alpha_link: float = 1.0  # α coefficient
    negatives_per_positive_train: int = 10
    negatives_per_positive_eval: int = 5

    # Training
    emb_lr: float = 1e-3
    link_lr: float = 1e-3
    target_batch_size: int = 50000

    # Data
    dataset: str  # required — no default
    data_dir: str = "data/"
    is_directed: bool  # required — dataset-specific
    max_node_count: int  # required — node IDs are expected to be [0, max_node_count)

    # Split ratios (TGB-identical np.quantile split)
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    # Pre-generated negative files (TGB pickle format)
    val_negative_file: Optional[str] = None
    test_negative_file: Optional[str] = None

    # System
    use_gpu: bool = False
    seed: int = 42

    @field_validator("val_ratio")
    @classmethod
    def _check_split_ratios(cls, v, info):
        train = info.data.get("train_ratio", 0.70)
        if train <= 0:
            raise ValueError(f"train_ratio must be > 0, got {train}")
        if v <= 0:
            raise ValueError(f"val_ratio must be > 0, got {v}")
        if train + v >= 1.0:
            raise ValueError(
                f"train_ratio + val_ratio must be < 1.0, got {train} + {v} = {train + v}"
            )
        return v
