from typing import Iterable, Optional

import numpy as np
import torch

from tempest_emb.config import Config
from tempest_emb.data.negative_sampler import UniformNegativeSampler
from tempest_emb.models.embedding_store import EmbeddingStore
from tempest_emb.models.link_predictor import LinkPredictor
from tempest_emb.training.embedding_trainer import EmbeddingTrainer
from tempest_emb.training.link_pred_trainer import LinkPredTrainer
from tempest_emb.types import Batch
from tempest_emb.utils.logging import Logger, load_checkpoint, save_checkpoint
from tempest_emb.walks.walk_generator import WalkGenerator


class Trainer:
    """Main loop orchestrator.

    Streaming protocol:
      - train(): ingest → walks → train_embedding → train_link_pred
      - val_or_test(): predict (via evaluator) → ingest → train_embedding only
    """

    def __init__(
        self,
        config: Config,
        node_feat: Optional[np.ndarray] = None,
        logger: Optional[Logger] = None,
    ):
        self.config = config
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )

        torch.manual_seed(config.seed)

        # Core modules
        self.embedding_store = EmbeddingStore(
            num_nodes=config.max_node_count,
            d_emb=config.d_emb,
            node_feat=node_feat,
        ).to(self.device)

        self.link_predictor = LinkPredictor(
            d_emb=config.d_emb,
            d_hidden=config.d_hidden_link,
        ).to(self.device)

        self.walk_gen = WalkGenerator(config)

        # Optimizers
        self.optimizer_emb = torch.optim.Adam(
            self.embedding_store.parameters(),
            lr=config.emb_lr,
        )
        self.optimizer_link = torch.optim.Adam(
            self.link_predictor.parameters(),
            lr=config.link_lr,
        )

        # Sub-trainers
        self.embedding_trainer = EmbeddingTrainer(
            embedding_store=self.embedding_store,
            optimizer_emb=self.optimizer_emb,
            config=config,
            device=self.device,
        )
        self.link_pred_trainer = LinkPredTrainer(
            embedding_store=self.embedding_store,
            link_predictor=self.link_predictor,
            optimizer_link=self.optimizer_link,
            device=self.device,
        )

        self.neg_sampler_train = UniformNegativeSampler(
            num_nodes=config.max_node_count,
            num_neg_per_pos=config.negatives_per_positive_train,
        )

        self.logger = logger or Logger(use_wandb=False)
        self.batch_idx = 0

    # ------------------------------------------------------------------ #
    # Phases
    # ------------------------------------------------------------------ #

    def train(self, batches: Iterable[Batch]) -> None:
        """Training phase.

        For each batch:
          1. Ingest into Tempest.
          2. Generate walks.
          3. Train embeddings (alignment + uniformity).
          4. Train link predictor (BCE on positives + sampled negatives).
        """
        self.link_predictor.train()
        for batch in batches:
            self.walk_gen.add_edges(batch.src, batch.tgt, batch.ts)

            walks = self.walk_gen.generate()
            l_align, l_uniform, l_total_emb = self.embedding_trainer.step(
                walks, batch.t_max
            )

            neg_src, neg_tgt = self.neg_sampler_train.sample(batch)
            l_link = self.link_pred_trainer.step(batch, neg_src, neg_tgt)

            self.batch_idx += 1
            self.logger.log(
                {
                    "train/align": l_align,
                    "train/uniform": l_uniform,
                    "train/emb_total": l_total_emb,
                    "train/link": l_link,
                },
                step=self.batch_idx,
            )

    def val_or_test(
        self,
        batches: Iterable[Batch],
        evaluator=None,
        phase: str = "val",
    ) -> Optional[float]:
        """Val/test phase with streaming evaluation.

        For each batch:
          1. (If evaluator provided) rank positives against eval negatives.
          2. Ingest into Tempest.
          3. Update embeddings (no link predictor training during val/test).

        Returns:
            Aggregate MRR if evaluator is provided, else None.
        """
        self.link_predictor.eval()
        total_rr = 0.0
        total_n = 0

        for batch in batches:
            if evaluator is not None:
                rr_sum, n = evaluator.evaluate_batch(batch)
                total_rr += rr_sum
                total_n += n

            self.walk_gen.add_edges(batch.src, batch.tgt, batch.ts)

            walks = self.walk_gen.generate()
            self.embedding_trainer.step(walks, batch.t_max)

            self.batch_idx += 1

        if evaluator is not None:
            mrr = total_rr / max(total_n, 1)
            self.logger.log({f"{phase}/mrr": mrr}, step=self.batch_idx)
            return mrr
        return None

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        save_checkpoint(
            path=path,
            config=self.config,
            batch_idx=self.batch_idx,
            embedding_store=self.embedding_store,
            link_predictor=self.link_predictor,
            optimizer_emb=self.optimizer_emb,
            optimizer_link=self.optimizer_link,
        )

    def load(self, path: str) -> None:
        self.batch_idx = load_checkpoint(
            path=path,
            embedding_store=self.embedding_store,
            link_predictor=self.link_predictor,
            optimizer_emb=self.optimizer_emb,
            optimizer_link=self.optimizer_link,
        )
