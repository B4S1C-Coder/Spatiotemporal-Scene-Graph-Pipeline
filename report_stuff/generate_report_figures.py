from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "font.size": 11,
        }
    )


def generate_loss_curve():
    rng = np.random.default_rng(7)
    epochs = np.arange(1, 31)

    train_box = 1.65 * np.exp(-epochs / 11.5) + 0.18 + rng.normal(0, 0.015, epochs.size)
    val_box = 1.75 * np.exp(-epochs / 10.0) + 0.24 + rng.normal(0, 0.02, epochs.size)
    train_cls = 1.25 * np.exp(-epochs / 10.5) + 0.10 + rng.normal(0, 0.012, epochs.size)
    val_cls = 1.35 * np.exp(-epochs / 9.5) + 0.13 + rng.normal(0, 0.016, epochs.size)

    fig, ax = plt.subplots()
    ax.plot(epochs, train_box, label="Train Box Loss", linewidth=2.2, color="#1d3557")
    ax.plot(epochs, val_box, label="Val Box Loss", linewidth=2.2, color="#457b9d")
    ax.plot(epochs, train_cls, label="Train Cls Loss", linewidth=2.2, color="#2a9d8f")
    ax.plot(epochs, val_cls, label="Val Cls Loss", linewidth=2.2, color="#e76f51")
    ax.set_title("Illustrative YOLOv8 Fine-Tuning Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.text(
        0.01,
        0.01,
        "Synthetic figure for report illustration only; not measured from this repository.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "illustrative_training_loss.png", dpi=220)
    plt.close(fig)


def generate_map_curve():
    rng = np.random.default_rng(11)
    epochs = np.arange(1, 31)

    map50 = 0.19 + 0.53 * (1 - np.exp(-epochs / 8.5)) + rng.normal(0, 0.008, epochs.size)
    map5095 = 0.08 + 0.39 * (1 - np.exp(-epochs / 9.5)) + rng.normal(0, 0.007, epochs.size)
    precision = 0.31 + 0.48 * (1 - np.exp(-epochs / 8.0)) + rng.normal(0, 0.01, epochs.size)
    recall = 0.24 + 0.44 * (1 - np.exp(-epochs / 7.0)) + rng.normal(0, 0.01, epochs.size)

    fig, ax = plt.subplots()
    ax.plot(epochs, map50, label="mAP@50", linewidth=2.2, color="#264653")
    ax.plot(epochs, map5095, label="mAP@50:95", linewidth=2.2, color="#e9c46a")
    ax.plot(epochs, precision, label="Precision", linewidth=2.2, color="#f4a261")
    ax.plot(epochs, recall, label="Recall", linewidth=2.2, color="#e76f51")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Illustrative Detection Quality Curves Across Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    ax.text(
        0.01,
        0.01,
        "Synthetic figure for report illustration only; replace with real validation metrics later.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "illustrative_detection_metrics.png", dpi=220)
    plt.close(fig)


def generate_latency_breakdown():
    stages = [
        "Detection",
        "Tracking",
        "Motion",
        "Events",
        "Graph\nBatching",
        "LLM Query\n(Separate Path)",
    ]
    latency_ms = [33, 7, 3, 5, 8, 180]
    colors = ["#1d3557", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]

    fig, ax = plt.subplots()
    bars = ax.bar(stages, latency_ms, color=colors)
    ax.set_title("Illustrative Runtime Cost by Pipeline Stage")
    ax.set_ylabel("Approximate Time (ms)")
    ax.set_xlabel("Stage")
    ax.bar_label(bars, padding=3, fmt="%.0f")
    ax.text(
        0.01,
        0.01,
        "Analytical illustration, not benchmarked timing. LLM query latency is off the video-ingestion critical path.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "illustrative_runtime_breakdown.png", dpi=220)
    plt.close(fig)


def main():
    _style()
    generate_loss_curve()
    generate_map_curve()
    generate_latency_breakdown()


if __name__ == "__main__":
    main()
