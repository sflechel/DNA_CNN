from torch.utils.data import DataLoader
import torch
from src.load_data import DNASeqDataset
from src.model import DNACNN
import logging
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import json
import numpy as np


def testing(args: argparse.Namespace) -> None:
    """Do predictions on testing dataset and output per-target ROC-AUC plot"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("training_log.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    test_dataset = DNASeqDataset(
        h5_filepath="data/processed/dataset_test.h5",
        augment_data=False,
        inner_size=args.inner_size,
        min_positives=args.min_positives,
        jitter=0,
    )

    batch_size: int = test_dataset.chunk_size * args.batch_size_multiplier

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    sequence, labels = next(iter(test_loader))

    logger.info(f"Sequence batch shape: {sequence.shape}")
    logger.info(f"Labels batch shape: {labels.shape}")
    logger.info(f"Labels mean: {labels.mean().item()}")

    device = device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Training on {device}")

    num_targets: int = test_dataset.num_targets
    seq_len: int = test_dataset.inner_size
    model: DNACNN = DNACNN(num_targets=num_targets, seq_len=seq_len).to(device)
    model.load_state_dict(torch.load(args.model_artifacts, map_location=device))

    correct_predictions = 0
    total_elements = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            true_batch_size = sequences.size(0)
            sequences = sequences.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).float()

            if sequences.size(2) > args.inner_size:
                start_idx = (sequences.size(2) - args.inner_size) // 2
                sequences = sequences[:, :, start_idx : start_idx + args.inner_size]
            logits = model(sequences)

            total_samples += true_batch_size

            probabilities = torch.sigmoid(logits.float())
            predictions = (probabilities >= 0.5).float()
            correct_predictions += (predictions == labels).sum().item()
            total_elements += predictions.numel()

            all_preds.append(probabilities)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
    filtered_preds = all_preds[:, test_dataset.mask]
    filtered_labels = all_labels[:, test_dataset.mask]

    prauc = average_precision_score(filtered_labels, filtered_preds, average="macro")
    auroc = roc_auc_score(filtered_labels, filtered_preds, average="macro")

    target_names = test_dataset.target_names
    logger.info(f"Macro PR-AUC: {prauc:.4f}, Macro ROC-AUC: {auroc:.4f}")

    # compute per-target ROC curves

    roc_data = []
    auc_scores = {}

    filtered_target_names = [
        name for i, name in enumerate(target_names) if test_dataset.mask[i]
    ]

    logger.info("Computing ROC-AUC per target...")
    num_filtered_targets = filtered_labels.shape[1]

    for i in range(num_filtered_targets):
        fpr, tpr, _ = roc_curve(filtered_labels[:, i], filtered_preds[:, i])
        auc = roc_auc_score(filtered_labels[:, i], filtered_preds[:, i])

        roc_data.append((fpr, tpr))

        target_name = (
            filtered_target_names[i]
            if i < len(filtered_target_names)
            else f"Target_{i}"
        )
        auc_scores[target_name] = auc

    median_auc = float(np.median(list(auc_scores.values())))
    logger.info(f"Median ROC-AUC: {median_auc:.4f}")

    with open(args.metrics_out, "w") as f:
        json.dump(auc_scores, f, indent=4)
    logger.info(f"Saved per-target metrics to {args.metrics_out}")

    # replicate the Zhou (2015) Figure 2a styling
    logger.info("Generating ROC plot...")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    for fpr, tpr in roc_data:
        ax.plot(fpr, tpr, color="black", alpha=0.12, linewidth=0.8)

    ax.set_xlim((-0.03, 1.03))
    ax.set_ylim((-0.03, 1.03))
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])

    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.set_title("Transcription factors", fontsize=12)

    # Despine: Remove top and right borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Tick marks pointing outward
    ax.tick_params(axis="both", which="both", direction="out", top=False, right=False)

    plt.tight_layout()

    plot_path = args.plot_out
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved ROC plot to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DNACNN and generate ROC plots.")

    parser.add_argument(
        "model_artifacts",
        type=str,
        default="output/checkpoints/best_model.pth",
        help="Path to saved model weights (.pt / .pth)",
    )
    parser.add_argument(
        "--inner_size", type=int, default=1000, help="Sequence length expected by model"
    )
    parser.add_argument(
        "--min_positives",
        type=int,
        default=1,
        help="Minimum positive samples to keep a target",
    )
    parser.add_argument(
        "--batch_size_multiplier", type=int, default=1, help="Multiplier for chunk_size"
    )
    parser.add_argument(
        "--num_workers", type=int, default=16, help="Number of DataLoader workers"
    )

    parser.add_argument(
        "--plot_out",
        type=str,
        default="output/plots/roc_auc_plot.png",
        help="Output path for ROC plot",
    )
    parser.add_argument(
        "--metrics_out",
        type=str,
        default="output/logs/test_metrics.json",
        help="Output path for metrics JSON",
    )

    args = parser.parse_args()

    testing(args)
