from collections import OrderedDict
from typing import Callable, Iterable, Tuple
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import List

# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def _quick_fit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    epochs: int,
    lr: float = 1e-3,
    device: str | torch.device,
    loss_fn: Callable,   # swap for CE in classification
    graph_loss: bool = True,
) -> float:
    """Fast, noisy training loop; returns best validation loss."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_test = float("inf")

    train_losses_epoch = []
    test_losses_epoch = []

    for _ in range(epochs):
        model.train()
        acc_loss, acc_samples = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            acc_loss += loss.item()
            acc_samples += yb.size(0)
        train_losses_epoch.append(acc_loss / acc_samples)

        # --- Evaluation ---
        model.eval()
        acc_loss, acc_samples = 0.0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = loss_fn(model(xb), yb)
                acc_loss += loss.item()
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
    return width // 2


# --------------------------------------------------------------------------- #
#  Unified search                                                             #
# --------------------------------------------------------------------------- #
def find_minimum_and_underperforming(
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    model_ctor: Callable[[int], nn.Module],          # takes *one* hidden width
    width_schedule: Iterable[int] = (2, 4, 8, 16, 32, 64, 128, 256),  # Log-spaced widths
    epochs_per_model: int = 20,
    plateau_tol: float = 0.01,
    underperf_tol: float = 1.25,
    shrink_rule: Callable[[int], int] = _halve_int,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    loss_fn: Callable,
    graph_train_loss_trends: bool = True,
    graph_test_loss_trends: bool = True,
    graph_best_test_loss_vs_width: bool = True,
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

    # ---------- 1) coarse sweep to find the plateau (minimum-performing) ---------- #
    for w in tqdm(width_schedule, desc="Width sweep", disable=True):
        best_test_loss, train_losses, test_losses = _quick_fit(
            model_ctor(w),
            train_loader,
            test_loader,
            epochs=epochs_per_model,
            device=device,
            loss_fn=loss_fn,
            graph_loss=False,
        )
        print(f"Width: {w}, Test loss: {best_test_loss}")
        results[w] = best_test_loss
        train_losses_trends.append(train_losses)
        test_losses_trends.append(test_losses)
        best_test_losses.append(best_test_loss)

        if prev_loss is not None:
            rel_improv = (prev_loss - best_test_loss) / prev_loss
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
    w_to_eval = shrink_rule(minimum_w)
    while under_w is None and w_to_eval >= 1:
        if w_to_eval not in results:            # train only once
            best_test_loss, train_losses, test_losses = _quick_fit(
                model_ctor(w_to_eval),
                train_loader,
                test_loader,
                epochs=epochs_per_model,
                device=device,
                loss_fn=loss_fn,
                graph_loss=False,
            )
            results[w_to_eval] = best_test_loss
            train_losses_trends.append(train_losses)
            test_losses_trends.append(test_losses)
            best_test_losses.append(best_test_loss)
        else:
            best_test_loss = results[w_to_eval]

        if best_test_loss > minimum_loss * underperf_tol:
            under_w = w_to_eval
            break

        w_to_eval = shrink_rule(w_to_eval)

    # ---------- 4) graph results ---------- #
    # put all graphs side by side in 3 columns
    n_graphs = sum([graph_train_loss_trends, graph_test_loss_trends, graph_best_test_loss_vs_width])
    if n_graphs >= 1:
        fig, axs = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(18, 5),
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
        axs[ax_idx].set_title(f"Train loss trends for {model_ctor.__name__}")
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
        axs[ax_idx].set_title(f"Test loss trends for {model_ctor.__name__}")
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
        axs[ax_idx].set_title(f"Best test loss vs. width for {model_ctor.__name__}")
        axs[ax_idx].set_xscale('log')  # Log scale for width
        axs[ax_idx].set_yscale('log')
        axs[ax_idx].legend()
        ax_idx += 1

    if n_graphs >= 1:
        plt.tight_layout()
        plt.show()

    return minimum_w, (under_w or w_to_eval)



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
    from my_datasets import get_all_datasets, _DATASET_TASK_TYPE
    for dataset_name, (train_loader, test_loader) in get_all_datasets(batch_size=32, test_size=0.2)[:1]:
        task_type = _DATASET_TASK_TYPE[dataset_name]
        n_features = train_loader.dataset.tensors[0].shape[1]
        minimum_w, under_w = find_minimum_and_underperforming(
            train_loader,
            test_loader,
            model_ctor=lambda w: TwoLayerMLP(n_features, 1, w),
            loss_fn=F.mse_loss if task_type == "regression" else F.cross_entropy,
            graph_train_loss_trends=True,
            graph_test_loss_trends=True,
            graph_best_test_loss_vs_width=True,
        )
        print(f"{dataset_name}: minimum-performing width : {minimum_w}, under-performing width   : {under_w}")
