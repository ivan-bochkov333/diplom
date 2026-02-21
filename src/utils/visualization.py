"""
Visualization utilities: trajectory plots, error curves, attention maps.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_trajectory_3d(gt_xyz, pred_xyz, title="Trajectory", save_path=None):
    """
    3D plot of ground-truth vs predicted camera trajectory.
    gt_xyz, pred_xyz: (N, 3) numpy arrays.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2],
            "b-o", markersize=2, label="Ground Truth", alpha=0.7)
    ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2],
            "r-o", markersize=2, label="Predicted", alpha=0.7)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        return fig


def plot_cumulative_error(errors, label, xlabel="Error", title="Cumulative Error Distribution",
                          save_path=None):
    """Plot cumulative distribution of errors."""
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sorted_errors, cdf, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction of frames")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        return fig


def plot_comparison_curves(errors_dict, xlabel="Error", title="Model Comparison",
                           save_path=None):
    """
    Plot multiple cumulative error curves for model comparison.
    errors_dict: {"model_name": np.array of errors}
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, errors in errors_dict.items():
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax.plot(sorted_errors, cdf, label=name)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction of frames")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        return fig


def plot_attention_map(image, attention_weights, title="Attention Map", save_path=None):
    """
    Overlay attention weights on an image.
    image: (H, W, 3) numpy array in [0, 1].
    attention_weights: (h, w) numpy array.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(image)
    att_resized = np.array(
        __import__("PIL").Image.fromarray(
            (attention_weights * 255).astype(np.uint8)
        ).resize((image.shape[1], image.shape[0]))
    ) / 255.0
    axes[1].imshow(att_resized, cmap="jet", alpha=0.5)
    axes[1].set_title(title)
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        return fig
