import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, Optional, Dict, Any, List, Tuple
from copy import deepcopy


class Trainer:
    """
    Consolidated training utility with optional action callback at a specified epoch,
    warmup+cosine scheduling, deterministic loaders, and snapshot/restore helpers.

    Typical usage:
        trainer = Trainer(model, loss_fn, device, lr=..., epochs=..., warmup_epochs=..., use_cosine=True)
        result = trainer.fit(train_loader, test_loader, start_epoch=0, end_epoch=action_epoch)
        state = trainer.snapshot()

        # For treatment runs, re-create trainer with same config and load snapshot
        trainer2 = Trainer(model_template, loss_fn, device, lr=..., epochs=..., warmup_epochs=...)
        trainer2.load_snapshot(state)
        result2 = trainer2.fit(
            train_loader, test_loader,
            start_epoch=action_epoch, end_epoch=epochs,
            action_epoch=action_epoch,
            on_action=lambda ctx: ...,  # capture stats, split, add params
        )
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: str | torch.device,
        *,
        lr: float,
        epochs: int,
        warmup_epochs: int = 0,
        use_cosine: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.use_cosine = use_cosine

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = self._build_scheduler(self.optimizer, total_epochs=epochs, warmup_epochs=warmup_epochs, use_cosine=use_cosine)

        self.train_losses: List[float] = []
        self.test_losses: List[float] = []

    # ---------- Public API ----------

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        *,
        start_epoch: int = 0,
        end_epoch: Optional[int] = None,
        action_epoch: Optional[int] = None,
        on_action: Optional[Callable[[Dict[str, Any]], None]] = None,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """
        Train from start_epoch (inclusive) to end_epoch (exclusive). If action_epoch is provided and
        equals (epoch+1) during training, invoke on_action(context) after stepping the scheduler
        and before evaluation. The context dict contains: { 'trainer', 'epoch' }.
        """
        if end_epoch is None:
            end_epoch = self.epochs
        if end_epoch <= start_epoch:
            return {
                'train_losses_epoch': [],
                'test_losses_epoch': [],
                'best_test': float('inf'),
                'best_test_epoch': None,
            }

        if deterministic:
            train_loader = self._make_deterministic_loader(train_loader)
            test_loader = self._make_deterministic_loader(test_loader)

        best_test = float('inf')
        best_test_epoch: Optional[int] = None
        cur_train_losses: List[float] = []
        cur_test_losses: List[float] = []

        for epoch in range(start_epoch, end_epoch):
            # Train one epoch
            self.model.train()
            acc_loss, acc_samples = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(xb), yb)
                loss.backward()
                self.model.capture_stats()
                self.optimizer.step()
                acc_loss += float(loss.item()) * yb.size(0)
                acc_samples += int(yb.size(0))
            cur_train_losses.append(acc_loss / max(acc_samples, 1))

            # Step the scheduler at end of epoch
            if self.scheduler is not None:
                self.scheduler.step()

            # Optional action callback at this epoch boundary (after sched step, before eval)
            if action_epoch is not None and on_action is not None and (epoch + 1) == action_epoch:
                ctx = {'trainer': self, 'epoch': epoch + 1}
                on_action(ctx)

            # Evaluate
            self.model.eval()
            acc_loss, acc_samples = 0.0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    l = self.loss_fn(self.model(xb), yb)
                    acc_loss += float(l.item()) * yb.size(0)
                    acc_samples += int(yb.size(0))
            avg_test_loss = acc_loss / max(acc_samples, 1)
            cur_test_losses.append(avg_test_loss)
            if avg_test_loss < best_test:
                best_test = avg_test_loss
                best_test_epoch = epoch + 1

        self.train_losses.extend(cur_train_losses)
        self.test_losses.extend(cur_test_losses)

        return {
            'train_losses_epoch': cur_train_losses,
            'test_losses_epoch': cur_test_losses,
        }

    def evaluate(self, loader: DataLoader) -> float:
        """Average per-sample loss on the given loader."""
        self.model.eval()
        acc_loss, acc_samples = 0.0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                l = self.loss_fn(self.model(xb), yb)
                acc_loss += float(l.item()) * yb.size(0)
                acc_samples += int(yb.size(0))
        return acc_loss / max(acc_samples, 1)

    def snapshot(self) -> Dict[str, Any]:
        """Return a deep-copied snapshot of model/optimizer/scheduler state."""
        snap: Dict[str, Any] = {
            'model_state': deepcopy(self.model.state_dict()),
            'opt_state': deepcopy(self.optimizer.state_dict()),
            'sched_state': deepcopy(self.scheduler.state_dict() if self.scheduler is not None else {}),
            'config': {
                'lr': self.lr,
                'epochs': self.epochs,
                'warmup_epochs': self.warmup_epochs,
                'use_cosine': self.use_cosine,
            }
        }
        return snap

    def load_snapshot(self, state: Dict[str, Any]) -> None:
        """Load a snapshot; assumes same model architecture and optimizer type."""
        self.model.load_state_dict(state['model_state'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer.load_state_dict(state['opt_state'])
        self.scheduler = self._build_scheduler(self.optimizer, total_epochs=self.epochs, warmup_epochs=self.warmup_epochs, use_cosine=self.use_cosine)
        sched_state = state.get('sched_state', None)
        if self.scheduler is not None and sched_state:
            self.scheduler.load_state_dict(sched_state)

    def add_new_params_follow_scheduler(self, new_params: List[nn.Parameter], base_lr: Optional[float] = None) -> None:
        """Add new parameters to the optimizer and extend scheduler base_lrs recursively."""
        if not new_params:
            return
        if base_lr is None:
            base_lr = self.optimizer.param_groups[0].get('lr', self.lr)
        self.optimizer.add_param_group({'params': new_params, 'lr': base_lr})
        if self.scheduler is not None:
            self._append_base_lrs_recursive(self.scheduler, base_lr)


    def get_train_losses(self) -> List[float]:
        return self.train_losses

    def get_test_losses(self) -> List[float]:
        return self.test_losses

    def graph_train_test_losses(self, title: str = 'Train and Test Losses', save_path: str | None = None) -> None:
        import matplotlib.pyplot as plt
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


    # ---------- Internal helpers ----------

    @staticmethod
    def _make_deterministic_loader(dl: DataLoader) -> DataLoader:
        return DataLoader(dl.dataset, batch_size=dl.batch_size, shuffle=False)

    @staticmethod
    def _build_scheduler(
        optimizer: torch.optim.Optimizer,
        *,
        total_epochs: int,
        warmup_epochs: int,
        use_cosine: bool,
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        if not use_cosine:
            return None
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs))
            return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        else:
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs))

    @staticmethod
    def _append_base_lrs_recursive(sched: torch.optim.lr_scheduler._LRScheduler, lr: float) -> None:
        # Append to this scheduler if it tracks base_lrs
        if hasattr(sched, 'base_lrs') and isinstance(sched.base_lrs, list):
            sched.base_lrs.append(lr)
        # Handle nested schedulers inside SequentialLR via public or private attribute
        inner = getattr(sched, 'schedulers', None)
        if inner is None:
            inner = getattr(sched, '_schedulers', None)
        if inner is not None:
            for sch in inner:
                Trainer._append_base_lrs_recursive(sch, lr)


