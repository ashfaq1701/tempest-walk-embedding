# Tempest Walk-First Embedding

A streaming, Tempest-native node embedding system for temporal graphs. Learns node embeddings from causality-preserving backward random walks using an alignment + uniformity objective (no negative sampling for embeddings), and trains a lightweight MLP for link prediction.

For the full design rationale, math, and pipeline details, see [`design_document.md`](design_document.md).

## Overview

```
Batch (src, tgt, t)
   ↓
Tempest backward walks
   ↓
Alignment loss   (pull walk-connected nodes together)
Uniformity loss  (push batch nodes apart)
   ↓
Embedding update (E_target, E_context)
   ↓
Link-prediction MLP (BCE on positives + sampled negatives)
   ↓
Streaming MRR evaluation
```

Key ideas:

- **Two embedding tables.** `E_target` ("what is this node?") and `E_context` ("what context does this node provide?") decouple identity and contextual roles.
- **No explicit negatives for embedding training.** Walk-based positives + uniformity replaces contradictory negative sampling.
- **Vectorized loss on a `[W, L-1]` walk grid.** Single batched cosine similarity per step.
- **Streaming evaluation.** Each batch is evaluated before being absorbed into the model (TGB-style protocol).
- **Optional node features** are concatenated into the alignment computation; link prediction uses embeddings only.

## Repository Layout

```
tempest_emb/
  config.py              # Pydantic Config (all hyperparameters)
  data/                  # Dataset loading, batching, negative sampling
  walks/                 # Tempest walk generator wrapper
  models/                # EmbeddingStore, LinkPredictor
  losses/                # Alignment, uniformity, link prediction
  training/              # EmbeddingTrainer, LinkPredTrainer, Trainer
  evaluation/            # Streaming MRR Evaluator
  utils/                 # Logging
scripts/
  train.py               # CLI entry point
notebooks/
  analysis.ipynb         # Exploration / analysis
data/                    # Place datasets here
checkpoints/             # Model checkpoints
design_document.md       # Full design specification
```

## Data Format

Datasets follow the TGN-style layout under `data/` (or whatever `--data-dir` points to):

- `{dataset}.csv` — columns `u, i, ts, idx`
- `{dataset}_edges.npy` — edge features indexed by `idx`, shape `[E, d_edge]`
- `{dataset}_node.npy` — optional node features indexed by dense node ID, shape `[N, d_node]`

Node IDs must be dense integers in `[0, max_node_count)`. The data is split chronologically using `np.quantile` on per-edge timestamps (configurable via `--train-ratio` / `--val-ratio`, defaults 0.70 / 0.15). This split is byte-for-byte identical to both TGB and Neural Temporal Walks for matching ratios.

## Training

Run training via the CLI:

```bash
python scripts/train.py \
    --dataset ml_collegemessage \
    --max-node-count 1899 \
    --directed \
    --data-dir data/ \
    --use-gpu \
    --checkpoint checkpoints/collegemessage.pt
```

Required flags:

| Flag | Description |
|---|---|
| `--dataset` | Dataset name (matches the file prefix in `--data-dir`) |
| `--max-node-count` | Total node count; node IDs must be in `[0, max_node_count)` |
| `--directed` / `--undirected` | Graph directionality (mutually exclusive) |

Common optional flags:

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | `data/` | Directory containing the dataset files |
| `--use-gpu` | off | Enable CUDA if available |
| `--d-emb` | 128 | Embedding dimension |
| `--target-batch-size` | 50000 | Approximate edges per batch (timestamp-respecting) |
| `--emb-lr` | 1e-3 | Embedding optimizer learning rate |
| `--link-lr` | 1e-3 | Link-prediction optimizer learning rate |
| `--negatives-per-positive-train` | 10 | Random negatives per positive during training |
| `--negatives-per-positive-eval` | 5 | Random negatives per positive during evaluation |
| `--seed` | 42 | Random seed |
| `--checkpoint` | none | Path to save the final checkpoint |
| `--train-ratio` | 0.70 | Fraction of edges for training (TGB-identical split) |
| `--val-ratio` | 0.15 | Fraction of edges for validation (TGB-identical split) |
| `--val-negative-file` | none | Path to TGB-format val negatives pickle |
| `--test-negative-file` | none | Path to TGB-format test negatives pickle |

All other hyperparameters (walk length, walks per node, decay exponents, loss coefficients, etc.) live in [`tempest_emb/config.py`](tempest_emb/config.py) — see Section 16 of the design document for the full inventory.

## Evaluation

After training, validation and test phases run streaming MRR evaluation:

- For each positive edge, score it against its negatives and compute pessimistic rank.
- **Pessimistic ranking**: ties count against the positive (`rank = (neg_scores >= pos_score).sum() + 1`).
- Report `MRR = mean(1 / rank)` across all batch edges.

During val/test, embeddings continue updating (the embedding trainer runs after evaluation), but the link predictor is frozen.

### Negative sampling modes

By default, evaluation uses `negatives_per_positive_eval` uniformly random negatives per positive. To use TGB's pre-generated negatives instead, pass pickle files:

```bash
python scripts/train.py \
    --dataset tgbl-wiki \
    --max-node-count 9228 \
    --undirected \
    --val-negative-file data/tgbl-wiki_val_ns.pkl \
    --test-negative-file data/tgbl-wiki_test_ns.pkl
```

The pickle format is a dict `{(src, dst, ts): np.array([neg_dst_1, ..., neg_dst_N])}` — the standard TGB negative sampler output. Val and test can use different modes independently (e.g. file-backed val with on-the-fly test). When using TGB pickles, the split must match TGB's (which it does by default with `--train-ratio 0.70 --val-ratio 0.15`).

## Checkpoints

`Trainer.save(path)` writes a `torch.save` bundle containing:

```python
{
    "embedding_store":  embedding_store.state_dict(),
    "link_predictor":   link_predictor.state_dict(),
    "optimizer_emb":    optimizer_emb.state_dict(),
    "optimizer_link":   optimizer_link.state_dict(),
    "batch_number":     current_batch,
    "config":           config.dict(),
}
```

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- Pydantic
- [temporal-random-walk](https://github.com/) — Tempest temporal random walk library

## License

See repository for license information.
