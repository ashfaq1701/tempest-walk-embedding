from pathlib import Path
from typing import Any, Dict, Optional

import torch

from tempest_emb.config import Config


class Logger:
    """Metrics logging with optional W&B integration."""

    def __init__(self, use_wandb: bool = False, wandb_project: Optional[str] = None,
                 wandb_config: Optional[Dict[str, Any]] = None):
        self.use_wandb = use_wandb
        self.run = None
        if use_wandb:
            import wandb
            self.run = wandb.init(project=wandb_project, config=wandb_config)

    def log(self, metrics: Dict[str, float], step: int) -> None:
        parts = [f"[step {step}]"] + [f"{k}={v:.6f}" for k, v in metrics.items()]
        print("  ".join(parts))
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.use_wandb:
            import wandb
            wandb.finish()


def save_checkpoint(
    path: str,
    config: Config,
    batch_idx: int,
    embedding_store: torch.nn.Module,
    link_predictor: torch.nn.Module,
    optimizer_emb: torch.optim.Optimizer,
    optimizer_link: torch.optim.Optimizer,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": config.model_dump(),
        "batch_idx": batch_idx,
        "embedding_store": embedding_store.state_dict(),
        "link_predictor": link_predictor.state_dict(),
        "optimizer_emb": optimizer_emb.state_dict(),
        "optimizer_link": optimizer_link.state_dict(),
    }, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    embedding_store: torch.nn.Module,
    link_predictor: torch.nn.Module,
    optimizer_emb: torch.optim.Optimizer,
    optimizer_link: torch.optim.Optimizer,
) -> int:
    ckpt = torch.load(path, weights_only=False)
    embedding_store.load_state_dict(ckpt["embedding_store"])
    link_predictor.load_state_dict(ckpt["link_predictor"])
    optimizer_emb.load_state_dict(ckpt["optimizer_emb"])
    optimizer_link.load_state_dict(ckpt["optimizer_link"])
    batch_idx = ckpt["batch_idx"]
    print(f"Checkpoint loaded: {path} (batch {batch_idx})")
    return batch_idx
