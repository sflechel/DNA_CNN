from math import nan
import time
from numpy import random
import torch
from torch.utils.data import DataLoader
from src.load_data import DNASeqDataset
from src.model import DNACNN
import torch.nn as nn
import torch.optim as optim
import logging
from pathlib import Path
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.tensorboard import SummaryWriter


def apply_weight_norm_constraint(model: nn.Module, lambda3: float) -> None:
    """Regularization preventing any weight from exceeding a certain value"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" not in name:
                continue
            if param.ndim == 2:
                l2norm = torch.norm(param, p=2, dim=1, keepdim=True)
            elif param.ndim == 3:
                l2norm = torch.norm(param, p=2, dim=[1, 2], keepdim=True)
            else:
                continue
            violators = l2norm > lambda3
            if violators.any():
                scale = lambda3 / (l2norm + 1e-8)
                param.mul_(torch.where(violators, scale, 1.0))


def gpu_side_data_augmentation(sequences: torch.Tensor, sequence_size: int) -> None:
    """Randomly shift the sequence trained on inside the larger window, and apply reverse-complement data augmentation"""
    if sequences.size(2) > sequence_size:
        max_offset = sequences.size(2) - sequence_size
        start_index = torch.randint(0, max_offset + 1, (1,)).item()
        sequences = sequences[:, :, start_index : start_index + sequence_size]

    if random.random() > 0.5:
        sequences = torch.flip(sequences, dims=[1, 2])


def no_data_augmentation(sequences: torch.Tensor, inner_size: int) -> None:
    """Take inner window exactly at center of outer window"""
    if sequences.size(2) > inner_size:
        start_idx = (sequences.size(2) - inner_size) // 2
        sequences = sequences[:, :, start_idx : start_idx + inner_size]


def train(args: argparse.Namespace):
    """Train the neural network"""
    writer = SummaryWriter(log_dir="runs")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("training_log.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    # initialization
    training_dataset = DNASeqDataset(
        h5_filepath="data/processed/dataset_train.h5",
        augment_data=True,
        inner_size=args.inner_size,
        jitter=args.jitter,
        min_positives=args.min_positives,
    )
    validation_dataset = DNASeqDataset(
        h5_filepath="data/processed/dataset_validation.h5",
        augment_data=False,
        inner_size=args.inner_size,
        jitter=0,
        min_positives=args.min_positives,
    )

    batch_size: int = training_dataset.chunk_size * args.batch_size_multiplier

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    sequence, labels = next(iter(training_loader))

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

    num_targets: int = training_dataset.num_targets
    seq_len: int = training_dataset.inner_size
    model: DNACNN = DNACNN(num_targets=num_targets, seq_len=seq_len).to(device)

    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    # training loop
    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(args.num_epochs):
        t0: float = time.time()
        model.train()

        running_loss = torch.tensor(0.0, device=device)

        for sequences, labels in training_loader:
            sequences = sequences.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).float()

            gpu_side_data_augmentation(sequences, args.inner_size)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                predictions, hidden_activation = model.forward_return_hidden(sequences)

                bceloss = criterion(predictions, labels)
                h_loss = (
                    args.output_decay
                    * torch.norm(hidden_activation, p=1)
                    / sequences.size(0)
                )
                loss = bceloss + h_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            apply_weight_norm_constraint(model, args.neuron_norm_max)
            running_loss += loss.detach()

        train_time: float = time.time() - t0
        t1: float = time.time()
        avg_loss = nan
        if len(training_loader) != 0:
            avg_loss = running_loss / len(training_loader)

        # model validation
        model.eval()
        validation_loss = 0.0
        correct_predictions = 0
        total_elements = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for validation_sequences, validation_labels in validation_loader:
                true_batch_size = validation_sequences.size(0)
                validation_sequences = validation_sequences.to(
                    device, non_blocking=True
                ).float()
                validation_labels = validation_labels.to(
                    device, non_blocking=True
                ).float()

                no_data_augmentation(validation_sequences, args.inner_size)

                with torch.amp.autocast("cuda"):
                    validation_logits = model(validation_sequences)
                    batch_loss = criterion(validation_logits, validation_labels)

                validation_loss += batch_loss.item() * validation_sequences.size(0)
                total_samples += true_batch_size

                probabilities = torch.sigmoid(validation_logits.float())
                validation_predictions = (probabilities >= 0.5).float()
                correct_predictions += (
                    (validation_predictions == validation_labels).sum().item()
                )
                total_elements += validation_predictions.numel()

                all_preds.append(probabilities)
                all_labels.append(validation_labels)

        # training statistics
        avg_validation_loss = validation_loss / total_samples
        scheduler.step(avg_validation_loss)
        validation_accuracy = correct_predictions / total_elements

        all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
        filtered_preds = all_preds[:, validation_dataset.mask]
        filtered_labels = all_labels[:, validation_dataset.mask]

        val_prauc = average_precision_score(
            filtered_labels, filtered_preds, average="macro"
        )
        val_auroc = roc_auc_score(filtered_labels, filtered_preds, average="macro")
        val_time: float = time.time() - t1

        if avg_validation_loss < best_val_loss:
            best_val_loss = avg_validation_loss
            patience_counter = 0
            checkpoint_path: Path = Path("output/checkpoints/best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"New best model found. Saving at {checkpoint_path}")
        else:
            patience_counter += 1
        if patience_counter >= 6:
            logger.info("No progress is being made, aborting")
            break

        # logging
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Loss/val", avg_validation_loss, epoch)
        writer.add_scalar("Metrics/Val_ROC_AUC", val_auroc, epoch)
        writer.add_scalar("Metrics/Val_PR_AUC", val_prauc, epoch)
        logger.info(
            f"At epoch {epoch + 1} of {args.num_epochs}: "
            f"Training loss: {avg_loss:.4f} | Validation loss: {avg_validation_loss:.4f} | "
            f"Validation accuracy: {validation_accuracy * 100:.2f}% |  Validation ROC-AUC: {val_auroc:.4f} | "
            f"Validation PR-AUC: {val_prauc:.4f} | "
            f"Training time: {train_time:.1f}, Validation time: {val_time:.1f}"
        )
    writer.close()
