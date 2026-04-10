# Tempest Walk-First Embedding System — Design Document v2

---

## 1. Goal

Build a fast, streaming, Tempest-native node embedding system that:

- Learns node embeddings from causality-preserving backward walks
- Uses alignment + uniformity loss (no explicit negatives for embedding training)
- Updates embeddings incrementally per batch
- Supports link prediction via a lightweight MLP

---

## 2. Core Idea

Tempest generates backward temporal random walks. Each walk traces a target node's causal history into the past. We train embeddings using an alignment loss (pull walk-connected nodes together) balanced by a uniformity loss (push all batch nodes apart). No negative sampling is needed for embedding training. Link prediction is trained separately on batch edges using a dedicated MLP.

**Core principle:** Walks → vectorized alignment on [W, L-1] grid → uniformity on batch node embeddings → embedding updates → link prediction on embeddings.

---

## 3. Walk Structure

Tempest backward walks produce:

```
nodes:      [n0, n1, n2, n3, n4]
timestamps: [t0, t1, t2, t3, INT_MAX]   ← last is sentinel, ignored
```

The walk is backward in time. Node `n0` is the target (most recent). Timestamps are decreasing: `t0 > t1 > t2 > t3`.

**Edges (backward):**

```
n4 → n3 at time t3
n3 → n2 at time t2
n2 → n1 at time t1
n1 → n0 at time t0
```

**Edge features:** Array of shape `[L-1, d_edge]`. Not used in the v2 alignment loss — their information is learned indirectly through embedding updates over many batches.

---

## 4. Walk Generation

Walks are generated via `get_random_walks_and_times_for_last_batch(...)`:

- For **directed graphs**, walks start from **unique source nodes** of the last batch.
- For **undirected graphs**, walks start from the **sorted union of unique source and target nodes** of the last batch.

`is_directed` is passed to `TemporalRandomWalk` at construction. Each starting node gets `num_walks_per_node` walks.

---

## 5. Pipeline

```
Batch of edges (src, tgt, t) arrives
   ↓
Ingest into Tempest (update streaming graph)
   ↓
Generate backward walks (num_walks_per_node per starting node)
   ↓
EmbeddingTrainer.step(walks, t_max)
   → compute alignment loss on [W, L-1] grid
   → compute uniformity loss on batch node embeddings
   → update E_target, E_context via optimizer_emb
   ↓
LinkPredTrainer.step(batch, neg_src, neg_tgt)
   → embeddings frozen, only MLP_link gets gradients
   → update MLP_link via optimizer_link
   ↓
[On demand]
Evaluator.evaluate_batch(batch) → MRR
```

---

## 6. Node Representation

**Two embedding tables:**

```
E_target  : [max_node_count, d_emb]    ← "what is this node?" — identity representation
E_context : [max_node_count, d_emb]    ← "what context does this node provide?" — contextual role
```

The two-table design decouples identity and context roles. With a single shared table, a node's embedding receives competing gradients from its target role and context role, causing instability.

**Node features (optional):** If the dataset provides node attributes `x_node[u]` of dimension `d_node`, they are concatenated with embeddings during alignment loss computation. This is the primary usage of node features. Link prediction reads only the learned embeddings — node feature information is already encoded through embedding training.

Both tables are used:
- **During embedding training:** `E_target` for target nodes (position 0), `E_context` for context nodes (positions 1+). Node features concatenated when present. Uniformity operates on `E_target` only.
- **During link prediction:** Both `E_target` and `E_context` are inputs. No node features.

---

## 7. Node Initialization

All embedding slots are pre-allocated to `max_node_count` and initialized with **Xavier-uniform**. No dynamic growth, no neighbor-copy, no `W_init` projection.

Rationale: `W_init` projections have no gradient path under direct `.data` writes. First-neighbor copy is not reliably knowable at runtime (a new node can appear in multiple edges simultaneously, and both endpoints can be new). Pre-allocation with Xavier-uniform is simple and robust.

---

## 8. Why No Explicit Negatives

Walk-based training creates indirect positive pairs. A walk `A ← B ← C ← D` produces positives `(A,B)`, `(A,C)`, `(A,D)`. But a negative sampler only knows about direct edges — it may sample C as a negative for A because no direct edge `(A,C)` exists. This creates contradictory gradients.

The set of indirect positives is intractable — the transitive closure of walk connectivity can cover a large fraction of the graph. If indirect positives are intractable, so are indirect negatives.

**Solution:** Alignment pulls walk-connected pairs together. Uniformity pushes all batch node embeddings apart. No pair is ever labeled as negative. The walk-overlap contradiction vanishes.

---

## 9. Loss Functions

### 9.1 Alignment Loss

For each walk, compute weighted cosine similarity between the target (position 0) and every valid context position. Operates directly on the `[W, L-1]` grid.

**Temporal weight per position:**

```
w_k = (1/k) * (1 + t_max - t_{k-1})^(-β)
```

- `1/k` — positional decay: distant nodes contribute less.
- `(1 + Δt)^(-β)` — temporal decay (power-law): stale interactions contribute less.

**Loss:**

```
L_align = (1/P) * Σ_{w,k} w_k * (1 - cos_sim(E_target[n0_w], E_context[n_k_w]))
```

Where `P` is the total number of valid pairs across all walks (clamped to min 1 to avoid division by zero when all walks have length 1).

When node features are present, they are concatenated before L2 normalization:

```
target_repr  = [E_target[n0]  | x_node[n0] ]     # [W, d_emb + d_node]
context_repr = [E_context[n_k] | x_node[n_k]]     # [W, L-1, d_emb + d_node]
```

### 9.2 Uniformity Loss

Pushes apart the embeddings of all unique nodes appearing in the current batch's walks. Operates on `E_target` directly.

```
batch_nodes = unique nodes from all walks (excluding padding -1)   # [U]
z = L2_normalize(E_target[batch_nodes])                            # [U, d]
sq_dist = 2 - 2 * (z @ z.T)                                       # [U, U]
L_uniform = (exp_sum - S) / (S * (S - 1))
```

Where `exp_sum = sum(exp(-t * sq_dist))` and `S = len(batch_nodes)`. The diagonal is subtracted analytically (`exp(0) = 1` for each of the S self-pairs) rather than materializing a boolean mask, avoiding a large allocation at the 20K cap.

Cap at 20K nodes with random subsampling if needed.

### 9.3 Link Prediction Loss

Standard binary cross-entropy over positive and negative edges:

```
p = sigmoid(MLP_link([E_target[u] | E_target[v] | E_context[u] | E_context[v]]))
L_link = BCE(p, y)
```

Negatives are sampled uniformly at random (pure random targets, no exclusion — collision rate is negligible for any reasonably sized graph). Only used here.

**Interleaved layout:** Positive and negative edges are arranged as `[pos_1, neg_1_1..neg_1_K, pos_2, neg_2_1..neg_2_K, ...]` via `cat + flatten` on `[B, 1+K]`. This keeps each positive grouped with its own negatives and makes MRR ranking trivially vectorizable.

### 9.4 Combined Batch Loss

```
L_total = L_align + η * L_uniform + α * L_link
```

Alignment is the anchor (coefficient 1.0). Two ratios to tune: `η` (uniformity strength) and `α` (link prediction strength).

---

## 10. Vectorized Alignment Computation

```python
# Step 1: Look up embeddings
e_target = E_target(walk_nodes[:, 0])                               # [W, d_emb]
e_context = E_context(walk_nodes[:, 1:])                            # [W, L-1, d_emb]

# Step 1b: Concatenate node features if present
if node_feat is not None:
    nf_target = node_feat[walk_nodes[:, 0]]                         # [W, d_node]
    nf_context = node_feat[walk_nodes[:, 1:]]                       # [W, L-1, d_node]
    e_target = torch.cat([e_target, nf_target], dim=-1)             # [W, d_emb + d_node]
    e_context = torch.cat([e_context, nf_context], dim=-1)          # [W, L-1, d_emb + d_node]

# Step 2: L2 normalize
e_target = F.normalize(e_target, dim=-1)
e_context = F.normalize(e_context, dim=-1)

# Step 3: Batched cosine similarity — one bmm
cos_sim = torch.bmm(
    e_target.unsqueeze(1),                                          # [W, 1, d]
    e_context.transpose(1, 2)                                       # [W, d, L-1]
).squeeze(1)                                                        # [W, L-1]

# Step 4: Temporal weights
delta_t = t_max - timestamps[:, :-1]                                # [W, L-1]
k = torch.arange(1, L, device=device).float()                      # [L-1]
w = (1.0 / k).unsqueeze(0) * (1 + delta_t).pow(-beta)              # [W, L-1]

# Step 5: Mask invalid positions
positions = torch.arange(1, L, device=device)
mask = positions.unsqueeze(0) < lens.unsqueeze(1)                   # [W, L-1]
w = w * mask

# Step 6: Loss
num_valid = mask.sum().clamp(min=1)
L_align = (w * (1 - cos_sim)).sum() / num_valid
```

---

## 11. Vectorized Uniformity Computation

```python
batch_node_ids = walk_nodes.unique()
batch_node_ids = batch_node_ids[batch_node_ids >= 0]                # exclude padding (-1)

if len(batch_node_ids) > cap:
    idx = torch.randperm(len(batch_node_ids), device=device)[:cap]
    batch_node_ids = batch_node_ids[idx]

S = len(batch_node_ids)
z = F.normalize(E_target(batch_node_ids), dim=-1)                   # [S, d]
gram = z @ z.T                                                      # [S, S]
sq_dist = 2 - 2 * gram
exp_mat = torch.exp(-t * sq_dist)

# Analytical diagonal subtraction: diagonal entries are exp(0) = 1, sum = S
L_uniform = (exp_mat.sum() - S) / (S * (S - 1))
```

---

## 12. Link Prediction Model (`MLP_link`)

Separate MLP operating on node embeddings only:

```
prob = sigmoid(MLP_link([ E_target(u) | E_target(v) | E_context(u) | E_context(v) ]))
```

Input dimension: `4 * d_emb`. Four separate embedding lookups — `E_target(u)`, `E_target(v)`, `E_context(u)`, `E_context(v)` — rather than fused. Clearer code, negligible overhead.

No edge features, no walk information, no node features.

---

## 13. Negative Sampling (Link Prediction Only)

Negatives are sampled via a stateless function `sample_negatives(batch, num_nodes, num_neg_per_pos)`. For each positive edge, the source is kept and the target is replaced with a uniformly random node from `[0, num_nodes)`. No history tracking or exclusion — collision with actual positives is negligible for any reasonably sized graph.

| Context | `num_neg_per_pos` |
|---|---|
| Link prediction training | `negatives_per_positive_train` (default 10) |
| Evaluation (val + test) | `negatives_per_positive_eval` (default 5) |

No negatives are used for embedding training.

---

## 14. MRR Evaluation

**Pessimistic ranking:** `rank = (neg_scores >= pos_score).sum() + 1`. Ties are counted against the positive edge.

For each positive edge `(u, v, t)`:
1. Score the positive and its K negatives through `MLP_link`.
2. Rank the positive among the K+1 scores.
3. `MRR = mean(1 / rank)` across all positive edges in the batch.

The interleaved layout `[pos, neg_1..neg_K, pos, neg_1..neg_K, ...]` makes this trivially vectorizable — reshape to `[B, 1+K]`, column 0 is the positive score, columns 1..K are negative scores.

---

## 15. Core Methods

### 15.1 `EmbeddingTrainer.step(walks, t_max)`

**Step 1 — Alignment.** Look up `E_target` for position 0, `E_context` for positions 1+. Concatenate node features if present. L2-normalize. Batched cosine similarity on [W, L-1] grid. Temporal weights. Mask. Weighted mean with clamp(min=1). (Section 10.)

**Step 2 — Uniformity.** Collect unique batch nodes (exclude padding -1). Subsample if > cap. Look up `E_target`. L2-normalize. Gram matrix. Analytical diagonal subtraction. (Section 11.)

**Step 3 — Loss:** `L_embedding = L_align + η * L_uniform`

**Step 4 — Update:** `optimizer_emb.step()` — updates `E_target` and `E_context`.

### 15.2 `LinkPredTrainer.step(batch, neg_src, neg_tgt)`

Embeddings are always frozen. Only `MLP_link` receives gradients.

```python
E_target.requires_grad_(False)
E_context.requires_grad_(False)

# Interleaved layout: [pos_1, neg_1_1..neg_1_K, pos_2, ...]
prob = sigmoid(MLP_link([E_target(u) | E_target(v) | E_context(u) | E_context(v)]))
loss = BCE(prob, labels)
loss.backward()
optimizer_link.step()
```

### 15.3 `Evaluator.evaluate_batch(batch) → MRR`

```python
# Score positive + negatives in interleaved layout
scores = MLP_link([E_target(u) | E_target(v) | E_context(u) | E_context(v)])
scores = scores.view(B, 1 + K)
pos_scores = scores[:, 0]
neg_scores = scores[:, 1:]

# Pessimistic ranking
rank = (neg_scores >= pos_scores.unsqueeze(1)).sum(dim=1) + 1
mrr = (1.0 / rank.float()).mean()
```

---

## 16. Parameter Inventory

### Learnable Parameters

| Component | Size |
|---|---|
| `E_target` | `[max_node_count, d_emb]` |
| `E_context` | `[max_node_count, d_emb]` |
| `MLP_link` | `(4 × d_emb) → d_hidden_link → 1` |

All initialized with Xavier-uniform.

### Hyperparameters

| Parameter | Description | Default |
|---|---|---|
| `d_emb` | Embedding dimension | 128 |
| `d_hidden_link` | MLP_link hidden dimension | 128 |
| `max_walk_len` | Maximum walk length | 10 |
| `num_walks_per_node` | Walks per starting node | 5 |
| `walk_bias` | Tempest walk bias | "Exponential" |
| `temporal_decay_exp` | Temporal decay exponent (β) | 0.5 |
| `eta_uniform` | Uniformity loss coefficient (η) | 1.0 |
| `uniformity_temperature` | Temperature in exp(-t * sq_dist) | 2.0 |
| `uniformity_cap` | Max nodes for gram matrix | 20000 |
| `alpha_link` | Link prediction loss coefficient (α) | 1.0 |
| `negatives_per_positive_train` | Random negatives per positive (training) | 10 |
| `negatives_per_positive_eval` | Random negatives per positive (evaluation) | 5 |
| `emb_lr` | Embedding optimizer learning rate | 1e-3 |
| `link_lr` | Link prediction optimizer learning rate | 1e-3 |
| `target_batch_size` | Approximate edges per batch | 50000 |
| `max_node_count` | Total node count (required) | — |
| `is_directed` | Graph directionality (required) | — |
| `dataset` | Dataset name (required) | — |
| `data_dir` | Dataset directory | "data/" |
| `use_gpu` | Use GPU if available | False |
| `seed` | Random seed | 42 |

---

## 17. Data Format

TGN-style custom datasets:

- `{dataset}.csv` — columns: `u, i, ts, idx`
- `{dataset}_edges.npy` — edge features indexed by `idx`, shape `[E, d_edge]`
- `{dataset}_node.npy` — node features indexed by dense node ID, shape `[N, d_node]`

Node feature dimension `d_node` is derived from the loaded array shape at runtime — not a config parameter.

---

## 18. Batching Strategy

Timestamp-respecting accumulation. All edges sharing a timestamp must appear in the same batch. Accumulate consecutive timestamp groups until hitting `target_batch_size`, never splitting within a timestamp. If a single timestamp exceeds `target_batch_size`, it becomes its own oversized batch.

---

## 19. Train/Val/Test Split

Chronological split, edge-count-guided, timestamp-respecting. Walk through unique timestamps accumulating edge counts, place split boundaries at 70%/85% of total edge count. All edges sharing a timestamp land in the same split.

```python
def chronological_split(timestamps, train_ratio=0.7, val_ratio=0.15):
    unique_ts, counts = np.unique(timestamps, return_counts=True)
    cumulative = np.cumsum(counts)
    total = cumulative[-1]

    train_idx = np.searchsorted(cumulative, total * train_ratio)
    val_idx = np.searchsorted(cumulative, total * (train_ratio + val_ratio))

    train_end_ts = unique_ts[train_idx]
    val_end_ts = unique_ts[val_idx]

    train_mask = timestamps <= train_end_ts
    val_mask = (timestamps > train_end_ts) & (timestamps <= val_end_ts)
    test_mask = timestamps > val_end_ts

    return train_mask, val_mask, test_mask
```

---

## 20. Evaluation Protocol

Following TGB protocol. Streaming evaluation — evaluate on each batch's edges before the model sees them as training data. During val/test, embeddings continue updating (EmbeddingTrainer runs after evaluation), link prediction does not train. MRR as primary metric. Transductive.

---

## 21. Class Structure

```
Trainer
  ├── EmbeddingTrainer     # alignment + uniformity, optimizer_emb
  ├── LinkPredTrainer      # BCE, optimizer_link (embeddings frozen)
  ├── Evaluator            # MRR prediction
  ├── EmbeddingStore       # E_target, E_context (pure storage, Xavier init)
  ├── LinkPredictor        # MLP_link
  └── WalkGenerator        # Tempest wrapper
```

`Trainer` owns all components and orchestrates:
- `train()` — main training loop over batches
- `val_or_test()` — streaming evaluation

### Optimizer Setup

Two separate optimizers:

```python
optimizer_emb = Adam(
    list(embedding_store.parameters()),
    lr=emb_lr
)

optimizer_link = Adam(
    link_predictor.parameters(),
    lr=link_lr
)
```

---

## 24. Checkpoint Format

Saved via `torch.save` with module `state_dict()`s:

```python
{
    "embedding_store": embedding_store.state_dict(),
    "link_predictor": link_predictor.state_dict(),
    "optimizer_emb": optimizer_emb.state_dict(),
    "optimizer_link": optimizer_link.state_dict(),
    "batch_number": current_batch,
    "config": config.dict(),
}
```

---

## 27. Summary

```
Tempest → Backward Walks → Embedding Lookup
                                ↓
              Alignment: bmm cosine similarity on [W, L-1] grid
              Weighted by w_k = (1/k) * (1+Δt)^(-β)
              Node features concatenated when present
              L_align = mean(w * (1 - cos_sim))
                                ↓
              Uniformity: gram matrix on batch node E_target
              Analytical diagonal subtraction
              L_uniform = (exp_sum - S) / (S * (S-1))
                                ↓
              L_embedding = L_align + η * L_uniform
                                ↓
              Link Prediction: MLP_link([Et(u) | Et(v) | Ec(u) | Ec(v)])
              Interleaved pos/neg layout
              L_link = BCE (uniform random negatives)
                                ↓
              L_total = L_embedding + α * L_link
                                ↓
              Evaluation (pessimistic MRR, streaming protocol)
```
