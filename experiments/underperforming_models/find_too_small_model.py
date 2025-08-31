from collections import OrderedDict
from typing import Callable, Iterable, Tuple
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def _quick_fit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    device: str | torch.device,
    loss_fn: Callable,   # swap for CE in classification
    graph_loss: bool = True,
    use_cosine_scheduler: bool = True,
    warmup_epochs: int = 0,
) -> float:
    """Fast, noisy training loop; returns best validation loss."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Add cosine learning rate scheduler with warmup
    if use_cosine_scheduler:
        if warmup_epochs > 0:
            # Use PyTorch's built-in schedulers for warmup + cosine annealing
            warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - warmup_epochs)
            scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        else:
            # Standard cosine annealing without warmup
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        scheduler = None
    
    best_test = float("inf")

    train_losses_epoch = []
    test_losses_epoch = []

    for epoch in range(epochs):
        model.train()
        acc_loss, acc_samples = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            acc_loss += loss.item() * yb.size(0)  # Fix #4: accumulate weighted loss
            acc_samples += yb.size(0)
        train_losses_epoch.append(acc_loss / acc_samples)

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

        # --- Evaluation ---
        model.eval()
        acc_loss, acc_samples = 0.0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = loss_fn(model(xb), yb)
                acc_loss += loss.item() * yb.size(0)  # Fix #4: accumulate weighted loss
                acc_samples += yb.size(0)
        avg_test_loss = acc_loss / acc_samples
        test_losses_epoch.append(avg_test_loss)

    best_test = min(test_losses_epoch)
    best_test_idx = test_losses_epoch.index(best_test)

    if graph_loss:
        plt.plot(train_losses_epoch, label="Train loss")
        plt.plot(test_losses_epoch, label="Test loss")
        plt.xlabel(f"epoch ({epochs} epochs)")
        plt.ylabel("per-sample loss")
        plt.title(f"Losses for {model.__class__.__name__}")
        plt.plot(best_test_idx, best_test, "ro", label="Best test loss")
        plt.legend()
        plt.show()

    return best_test, train_losses_epoch, test_losses_epoch


def _halve_int(width: int) -> int:
    """Shrink rule for a single scalar hidden width."""
    return max(width // 2, 1)  # Fix #2: prevent width 0


# --------------------------------------------------------------------------- #
#  Unified search                                                             #
# --------------------------------------------------------------------------- #
def find_minimum_and_underperforming(
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    model_ctor: Callable[[int], nn.Module],          # takes *one* hidden width
    width_schedule: Iterable[int] = (2, 4, 8, 16, 32, 64, 128, 256, 512),  # Log-spaced widths
    epochs_per_model: int = 20,
    plateau_tol: float = 0.01,
    underperf_tol: float = 1.25,
    shrink_rule: Callable[[int], int] = _halve_int,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    loss_fn: Callable,
    lr: float = 1e-3,
    use_cosine_scheduler: bool = True,
    warmup_epochs: int = 0,
    graph_train_loss_trends: bool = True,
    graph_test_loss_trends: bool = True,
    graph_best_test_loss_vs_width: bool = True,
    dataset_name: str = None,
    plot_save_path: str = None,
    result_save_path: str = None,
) -> Tuple[int, int]:
    """
    Discover (a) the minimum-performing width and (b) the next smaller, clearly
    under-performing width — without training any width twice.

    Returns
    -------
    minimum_width, under_width : int, int
    """
    results: "OrderedDict[int, float]" = OrderedDict()

    prev_w, prev_loss = None, None
    minimum_w, minimum_loss = None, None

    train_losses_trends: List[List[float]] = []
    test_losses_trends: List[List[float]] = []
    best_test_losses: List[float] = []

    # Fix #1: Division by zero protection
    EPS = 1e-12

    # if plateau_tol < 0, we will test all widths
    if plateau_tol <= 0:
        print(f"No plateau tolerance provided, will test all widths")

    # ---------- 1) coarse sweep to find the plateau (minimum-performing) ---------- #
    for w in tqdm(width_schedule, desc="Width sweep", disable=True):  # Fix #9: enable progress bar
        best_test_loss, train_losses, test_losses = _quick_fit(
            model_ctor(w),
            train_loader,
            test_loader,
            lr=lr,
            epochs=epochs_per_model,
            device=device,
            loss_fn=loss_fn,
            graph_loss=False,
            use_cosine_scheduler=use_cosine_scheduler,
            warmup_epochs=warmup_epochs,
        )
        print(f"{dataset_name}: Width: {w}, Test loss: {best_test_loss}")
        results[w] = best_test_loss
        train_losses_trends.append(train_losses)
        test_losses_trends.append(test_losses)
        best_test_losses.append(best_test_loss)

        if plateau_tol > 0:
            if prev_loss is not None:
                # Fix #1: Division by zero protection
                if prev_loss < EPS:
                    minimum_w, minimum_loss = prev_w, prev_loss
                    break
                rel_improv = (prev_loss - best_test_loss) / max(prev_loss, EPS)
                if rel_improv < plateau_tol:
                    minimum_w, minimum_loss = prev_w, prev_loss
                    break

        prev_w, prev_loss = w, best_test_loss

    if minimum_w is None:                       # never plateaued
        minimum_w, minimum_loss = prev_w, prev_loss

    # ---------- 2) find first clearly under-performing width ---------- #
    under_w = next(
        (w for w, loss in results.items()
         if w < minimum_w and loss > minimum_loss * underperf_tol),
        None,
    )

    # ---------- 3) keep shrinking if still not found ---------- #
    # w_to_eval = shrink_rule(minimum_w)
    # while under_w is None and w_to_eval >= 1:
    #     if w_to_eval not in results:            # train only once
    #         best_test_loss, train_losses, test_losses = _quick_fit(
    #             model_ctor(w_to_eval),
    #             train_loader,
    #             test_loader,
    #             epochs=epochs_per_model,
    #             lr=lr,
    #             device=device,
    #             loss_fn=loss_fn,
    #             graph_loss=False,
    #             use_cosine_scheduler=use_cosine_scheduler,
    #             warmup_epochs=warmup_epochs,
    #         )
    #         results[w_to_eval] = best_test_loss
    #         train_losses_trends.append(train_losses)
    #         test_losses_trends.append(test_losses)
    #         best_test_losses.append(best_test_loss)
    #     else:
    #         best_test_loss = results[w_to_eval]

    #     if best_test_loss > minimum_loss * underperf_tol:
    #         under_w = w_to_eval
    #         break

    #     w_to_eval = shrink_rule(w_to_eval)  # Fix #3: correct variable name
    # We will disable this for now
    if under_w is None:
        print(f"No under-performing width found according to the tolerance. Setting under_w to the smallest width.")
        under_w = min(results.keys())

    # ---------- 4) graph results ---------- #
    # put all graphs side by side in 3 columns
    n_graphs = sum([graph_train_loss_trends, graph_test_loss_trends, graph_best_test_loss_vs_width])
    
    # Fix #6: Guard plotting block
    if n_graphs >= 1:
        fig, axs = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(18, 5)
        )
        axs = axs.flatten()
        ax_idx = 0

        if graph_train_loss_trends:
            # Plot each width's train loss trend
            tested_widths = list(results.keys())
            for w, train_losses in zip(tested_widths, train_losses_trends):
                axs[ax_idx].plot(train_losses, label=f"Width: {w}")
            axs[ax_idx].set_xlabel(f"epoch ({epochs_per_model} epochs)")
            axs[ax_idx].set_ylabel("per-sample loss")
            axs[ax_idx].set_title(f"Train loss trends for {model_ctor.__name__} ({dataset_name} Dataset)")
            axs[ax_idx].set_yscale('log')
            axs[ax_idx].legend()
            ax_idx += 1

        if graph_test_loss_trends:
            # Plot each width's test loss trend
            tested_widths = list(results.keys())
            for w, test_losses in zip(tested_widths, test_losses_trends):
                axs[ax_idx].plot(test_losses, label=f"Width: {w}")
            axs[ax_idx].set_xlabel(f"epoch ({epochs_per_model} epochs)")
            axs[ax_idx].set_ylabel("per-sample loss")
            axs[ax_idx].set_title(f"Test loss trends for {model_ctor.__name__} ({dataset_name} Dataset)")
            axs[ax_idx].set_yscale('log')
            axs[ax_idx].legend()
            ax_idx += 1

        if graph_best_test_loss_vs_width:
            # Get the widths that were actually tested
            tested_widths = list(results.keys())
            tested_losses = [results[w] for w in tested_widths]
            
            axs[ax_idx].plot(tested_widths, tested_losses, label="Best test loss", marker='o')
            axs[ax_idx].set_xlabel("Width")
            axs[ax_idx].set_ylabel("Best test loss")
            axs[ax_idx].set_title(f"Best test loss vs. width for {model_ctor.__name__} ({dataset_name} Dataset)")
            axs[ax_idx].set_xscale('log')  # Log scale for width
            axs[ax_idx].set_yscale('log')
            axs[ax_idx].legend()
            ax_idx += 1

        # ---------- 5) save results ---------- #
        # Save Plots to file
        plt.tight_layout()
        if plot_save_path is not None:
            plt.savefig(plot_save_path)
        else:
            plt.show()

    # Save width vs best_test_loss results to csv file
    if result_save_path is not None:
        with open(result_save_path, "w") as f:
            f.write("width,best_test_loss\n")
            # Fix #8: Sort by width for better readability
            for w in sorted(results):
                f.write(f"{w},{results[w]}\n")

    return minimum_w, (under_w or w_to_eval)

def make_width_schedule(n_features: int,
                        max_width_cap: int = 1024) -> list[int]:
    """
    Powers-of-two up to max_width_cap.
    """
    widths = []
    w = 2
    while w <= max_width_cap:
        widths.append(w)
        w *= 2
    return widths

# Simple MLP with two hidden layers using the same width
class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, output_dim),
        )

    def forward(self, x):
        output = self.net(x)
        # Keep the output shape as is - my_datasets.py expects (N, 1) for regression
        return output

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import numpy as np
    from my_datasets import get_all_datasets, _DATASET_TASK_TYPE

    # HYPERPARAMETERS
    MAX_WIDTH_CAP = 512
    BATCH_SIZE = 128
    TEST_SIZE = 0.2
    LEARNING_RATE = 5e-3
    EPOCHS_PER_MODEL = 100
    PLATEAU_TOL = -1 # -1 means test all widths
    UNDERPERF_TOL = 1.25
    USE_COSINE_SCHEDULER = True
    WARMUP_EPOCHS = 10

    # results directory
    from datetime import datetime
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = Path(f"experiments/underperforming_models/{datetime_str}/")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save hyperparameters to file
    with open(RESULTS_DIR / "hyperparameters.txt", "w") as f:
        f.write(f"MAX_WIDTH_CAP: {MAX_WIDTH_CAP}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"TEST_SIZE: {TEST_SIZE}\n")
        f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
        f.write(f"EPOCHS_PER_MODEL: {EPOCHS_PER_MODEL}\n")
        f.write(f"PLATEAU_TOL: {PLATEAU_TOL}\n")
        f.write(f"UNDERPERF_TOL: {UNDERPERF_TOL}\n")
        f.write(f"USE_COSINE_SCHEDULER: {USE_COSINE_SCHEDULER}\n")
        f.write(f"WARMUP_EPOCHS: {WARMUP_EPOCHS}\n")

    for dataset_name, (train_loader, test_loader) in get_all_datasets(batch_size=BATCH_SIZE, test_size=TEST_SIZE)[3:5]:
        task_type = _DATASET_TASK_TYPE[dataset_name]
        n_features = train_loader.dataset.tensors[0].shape[1]
        
        # Fix #5: Determine correct output dimension for classification
        if task_type == "classification":
            # Get number of classes from the dataset
            y = train_loader.dataset.tensors[1]
            n_classes = int(y.max().item() + 1)
        else:
            n_classes = 1
            
        width_schedule = make_width_schedule(n_features, max_width_cap=MAX_WIDTH_CAP)
        print(f"{dataset_name}: n_features: {n_features}, n_classes: {n_classes}, width_schedule: {width_schedule}")
        minimum_w, under_w = find_minimum_and_underperforming(
            train_loader,
            test_loader,
            width_schedule=width_schedule,
            epochs_per_model=EPOCHS_PER_MODEL,
            plateau_tol=PLATEAU_TOL,
            underperf_tol=UNDERPERF_TOL,
            lr=LEARNING_RATE,
            use_cosine_scheduler=USE_COSINE_SCHEDULER,
            warmup_epochs=WARMUP_EPOCHS,
            model_ctor=lambda w: TwoLayerMLP(n_features, n_classes, w),  # Fix #5: use correct output_dim
            loss_fn=F.mse_loss if task_type == "regression" else F.cross_entropy,
            graph_train_loss_trends=True,
            graph_test_loss_trends=True,
            graph_best_test_loss_vs_width=True,
            dataset_name=dataset_name,
            plot_save_path=RESULTS_DIR / f"{dataset_name}_1_hidden_layer_model_size_stats.png",
            result_save_path=RESULTS_DIR / f"{dataset_name}_1_hidden_layer_model_width_vs_test_loss.csv",
        )
        print(f"{dataset_name}: minimum-performing width : {minimum_w}, under-performing width   : {under_w}")
