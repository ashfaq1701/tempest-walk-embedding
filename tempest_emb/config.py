from pydantic import BaseModel


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

    # System
    use_gpu: bool = False
    seed: int = 42
