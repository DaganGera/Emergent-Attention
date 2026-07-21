"""
Training engine for NCA-ViT experiments.

Handles:
  - AMP (mixed precision)
  - Gradient clipping
  - EMA (Exponential Moving Average)
  - Mixup / CutMix
  - W&B logging
  - Checkpoint save/resume
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from timm.utils import ModelEmaV3

import wandb

from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.data.augmentations import build_mixup
from src.utils.checkpoint import save_checkpoint, load_checkpoint, save_best_model


class Trainer:
    """
    End-to-end training loop.

    Args:
        model:          The model to train (NCAViT / HybridNCAViT / baseline)
        train_loader:   Training DataLoader
        val_loader:     Validation DataLoader
        cfg:            Hydra DictConfig (contains model, data, training sub-configs)
        device:         torch.device to train on
        output_dir:     Directory for checkpoints and logs
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg,
        device: torch.device,
        output_dir: str = "checkpoints",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir

        # cfg.training = entire default.yaml; nested keys:
        #   .optimizer, .scheduler, .training (hyperparams), .augmentation
        train_cfg = cfg.training.training   # the training: block in default.yaml
        aug_cfg = cfg.training.augmentation
        sched_cfg = cfg.training.scheduler

        # Optimizer and scheduler
        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)

        # Loss: use plain CrossEntropy when Mixup handles label smoothing,
        # else apply label smoothing here
        self.mixup_fn = build_mixup(
            mixup_alpha=aug_cfg.mixup_alpha,
            cutmix_alpha=aug_cfg.cutmix_alpha,
            mixup_prob=aug_cfg.mixup_prob,
            cutmix_prob=aug_cfg.cutmix_prob,
            num_classes=cfg.model.get("num_classes", 100),
            label_smoothing=train_cfg.label_smoothing,
        )
        if self.mixup_fn is not None:
            # Mixup already handles label smoothing → plain CE
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)

        # AMP
        self.use_amp = train_cfg.amp and device.type == "cuda"
        self._amp_device = device.type  # "cuda" or "cpu"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        # EMA
        self.ema_model: ModelEmaV3 | None = None
        if train_cfg.ema:
            self.ema_model = ModelEmaV3(model, decay=train_cfg.ema_decay, device=device)

        # Training state
        self.start_epoch = 0
        self.best_acc = 0.0
        self.total_epochs = sched_cfg.epochs
        self.grad_clip = train_cfg.grad_clip
        self.checkpoint_freq = train_cfg.checkpoint_freq
        self.eval_freq = train_cfg.eval_freq

        # Global step counter for W&B
        self.global_step = 0

    def train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Apply Mixup / CutMix
            if self.mixup_fn is not None:
                images, targets = self.mixup_fn(images, targets)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self._amp_device, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                ).item()
            else:
                grad_norm = 0.0

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # EMA update
            if self.ema_model is not None:
                self.ema_model.update(self.model)

            # Stats (use hard labels for accuracy even with Mixup)
            total_loss += loss.item() * images.size(0)
            if self.mixup_fn is None:
                correct += (outputs.argmax(1) == targets).sum().item()
                total += images.size(0)

            # W&B step log
            if wandb.run is not None:
                wandb.log({
                    "train/loss_step": loss.item(),
                    "train/grad_norm": grad_norm,
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "system/gpu_memory_MB": (
                        torch.cuda.max_memory_allocated() / 1e6
                        if self.device.type == "cuda" else 0
                    ),
                }, step=self.global_step)
            self.global_step += 1

        elapsed = time.time() - t0
        avg_loss = total_loss / len(self.train_loader.dataset)
        acc = (correct / total * 100) if total > 0 else 0.0
        return {"loss": avg_loss, "acc": acc, "time_s": elapsed}

    @torch.no_grad()
    def evaluate(self, use_ema: bool = True) -> dict:
        eval_model = (
            self.ema_model.module
            if (use_ema and self.ema_model is not None)
            else self.model
        )
        eval_model.eval()

        correct1 = correct5 = total = 0

        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self._amp_device, enabled=self.use_amp):
                outputs = eval_model(images)

            # Top-1
            correct1 += (outputs.argmax(1) == targets).sum().item()

            # Top-5
            top5 = outputs.topk(5, dim=1).indices
            correct5 += (top5 == targets.unsqueeze(1)).any(dim=1).sum().item()

            total += images.size(0)

        return {
            "top1": correct1 / total * 100,
            "top5": correct5 / total * 100,
        }

    def resume(self, checkpoint_path: str) -> None:
        """Resume training from a checkpoint."""
        state = load_checkpoint(checkpoint_path)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if self.ema_model is not None and "ema_model" in state:
            # Checkpoint stores ema_model.module.state_dict() (no "module." prefix),
            # so load into .module, not the EmaV3 wrapper itself.
            self.ema_model.module.load_state_dict(state["ema_model"])
        self.start_epoch = state["epoch"] + 1
        self.best_acc = state["best_acc"]
        print(f"Resumed from epoch {self.start_epoch}, best_acc={self.best_acc:.2f}%")

    def fit(self) -> None:
        """Full training loop."""
        for epoch in range(self.start_epoch, self.total_epochs):
            train_metrics = self.train_one_epoch(epoch)
            self.scheduler.step()
            # Release any fragmented CUDA memory before evaluation + checkpointing
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            log = {"epoch": epoch, "train/loss": train_metrics["loss"],
                   "train/acc": train_metrics["acc"], "train/lr": self.optimizer.param_groups[0]["lr"]}

            # Evaluate
            if (epoch + 1) % self.eval_freq == 0:
                val_metrics = self.evaluate()
                log["val/acc_top1"] = val_metrics["top1"]
                log["val/acc_top5"] = val_metrics["top5"]

                is_best = val_metrics["top1"] > self.best_acc
                if is_best:
                    self.best_acc = val_metrics["top1"]
                    # Save EMA model as best when available (matches evaluation)
                    best_model = self.ema_model.module if self.ema_model is not None else self.model
                    save_best_model(best_model, os.path.join(self.output_dir, "best.pt"))

                print(
                    f"Epoch [{epoch+1}/{self.total_epochs}]  "
                    f"loss={train_metrics['loss']:.4f}  "
                    f"val_top1={val_metrics['top1']:.2f}%  "
                    f"best={self.best_acc:.2f}%  "
                    f"({train_metrics['time_s']:.0f}s)"
                )
            else:
                print(
                    f"Epoch [{epoch+1}/{self.total_epochs}]  "
                    f"loss={train_metrics['loss']:.4f}  "
                    f"({train_metrics['time_s']:.0f}s)"
                )

            # Periodic checkpoint
            if (epoch + 1) % self.checkpoint_freq == 0:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    best_acc=self.best_acc,
                    path=os.path.join(self.output_dir, f"checkpoint_epoch{epoch+1:04d}.pt"),
                    ema_model=self.ema_model.module if self.ema_model else None,
                )
            # Always save latest
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                best_acc=self.best_acc,
                path=os.path.join(self.output_dir, "latest.pt"),
                ema_model=self.ema_model.module if self.ema_model else None,
            )

            if wandb.run is not None:
                wandb.log(log, step=self.global_step)

        print(f"Training complete. Best val acc: {self.best_acc:.2f}%")
