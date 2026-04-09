import torch
import torch.nn.functional as F


def uniformity_loss(
    walk_nodes: torch.Tensor,
    e_target: torch.nn.Embedding,
    temperature: float = 2.0,
    cap: int = 20000,
) -> torch.Tensor:
    """Uniformity loss on batch node E_target embeddings.

    Pushes apart embeddings of all unique nodes appearing in the current batch's walks
    via pairwise Gaussian repulsion on the unit hypersphere.

    Args:
        walk_nodes:  [W, L] node IDs from walks.
        e_target:    E_target embedding table.
        temperature: t in exp(-t * sq_dist).
        cap:         Max nodes for gram matrix (subsampled if exceeded).

    Returns:
        Scalar uniformity loss.
    """
    batch_node_ids = walk_nodes.unique()
    batch_node_ids = batch_node_ids[batch_node_ids >= 0]  # exclude padding (-1)

    if len(batch_node_ids) > cap:
        idx = torch.randperm(len(batch_node_ids), device=batch_node_ids.device)[:cap]
        batch_node_ids = batch_node_ids[idx]

    z = F.normalize(e_target(batch_node_ids), dim=-1)  # [S, d]
    gram = z @ z.T                                      # [S, S]
    sq_dist = 2 - 2 * gram
    exp_mat = torch.exp(-temperature * sq_dist)         # [S, S]
    # Diagonal is exp(-temperature * 0) = 1 at each slot (S entries total).
    # Subtract analytically instead of materializing a boolean-masked copy.
    S = z.shape[0]
    return (exp_mat.sum() - S) / (S * (S - 1))
