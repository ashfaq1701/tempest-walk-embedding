"""Entry point for training the Tempest walk-first embedding system."""

import argparse

import numpy as np
import torch

from tempest_emb.config import Config
from tempest_emb.data.dataset import create_batches, load_dataset
from tempest_emb.evaluation.evaluator import Evaluator
from tempest_emb.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Tempest walk-first embeddings")

    # Required dataset args
    p.add_argument("--dataset", type=str, required=True,
                   help="Dataset name (e.g. ml_collegemessage)")
    p.add_argument("--max-node-count", type=int, required=True,
                   help="Node IDs are expected to be in [0, max_node_count)")

    # Directed / undirected (mutually exclusive)
    direction = p.add_mutually_exclusive_group(required=True)
    direction.add_argument("--directed", dest="is_directed", action="store_true")
    direction.add_argument("--undirected", dest="is_directed", action="store_false")

    # Optional
    p.add_argument("--data-dir", type=str, default="data/")
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--d-emb", type=int, default=128)
    p.add_argument("--target-batch-size", type=int, default=50000)
    p.add_argument("--emb-lr", type=float, default=1e-3)
    p.add_argument("--link-lr", type=float, default=1e-3)
    p.add_argument("--negatives-per-positive-train", type=int, default=10)
    p.add_argument("--negatives-per-positive-eval", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to save checkpoint at end of run")

    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        dataset=args.dataset,
        data_dir=args.data_dir,
        max_node_count=args.max_node_count,
        is_directed=args.is_directed,
        use_gpu=args.use_gpu,
        d_emb=args.d_emb,
        target_batch_size=args.target_batch_size,
        emb_lr=args.emb_lr,
        link_lr=args.link_lr,
        negatives_per_positive_train=args.negatives_per_positive_train,
        negatives_per_positive_eval=args.negatives_per_positive_eval,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    print(f"Loading dataset: {config.dataset}")
    train_data, val_data, test_data, node_feat = load_dataset(config)

    if node_feat is not None:
        print(f"Node features: [{node_feat.shape[0]}, {node_feat.shape[1]}]")
    else:
        print("No node features")

    trainer = Trainer(config=config, node_feat=node_feat)
    evaluator = Evaluator(
        embedding_store=trainer.embedding_store,
        link_predictor=trainer.link_predictor,
        num_nodes=config.max_node_count,
        negatives_per_positive=config.negatives_per_positive_eval,
        device=trainer.device,
    )

    print(f"Device: {trainer.device}")

    # Training phase
    print("=== Training ===")
    train_batches = create_batches(train_data, config.target_batch_size)
    trainer.train(train_batches)

    # Validation phase (streaming eval)
    print("=== Validation ===")
    val_batches = create_batches(val_data, config.target_batch_size)
    val_mrr = trainer.val_or_test(val_batches, evaluator=evaluator, phase="val")
    print(f"Val MRR: {val_mrr:.4f}")

    # Test phase (streaming eval)
    print("=== Test ===")
    test_batches = create_batches(test_data, config.target_batch_size)
    test_mrr = trainer.val_or_test(test_batches, evaluator=evaluator, phase="test")
    print(f"Test MRR: {test_mrr:.4f}")

    if args.checkpoint is not None:
        trainer.save(args.checkpoint)

    trainer.logger.finish()


if __name__ == "__main__":
    main()
