from math import nan
import torch
from torch.utils.data import DataLoader
from src.load_data import DNASeqDataset
from src.model import DNACNN
import torch.nn as nn
import torch.optim as optim
import logging
from pathlib import Path
import argparse
import sklearn.metrics


def apply_weight_norm_constraint(model: DNACNN, lambda3: float) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" not in name:
                continue
            if param.ndim == 2:
                l2norm = torch.norm(param, p=2, dim=1, keepdim=True)
                clamped = torch.clamp(l2norm, max=lambda3)
                param.copy_(param * (clamped / (l2norm + 1e-8)))
            elif param.ndim == 3:
                l2norm = torch.norm(param, p=2, dim=[1, 2], keepdim=True)
                clamped = torch.clamp(l2norm, max=lambda3)
                param.copy_(param * clamped / (l2norm + 1e-8))


def train(args: argparse.Namespace):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("training_log.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    training_chroms = [f"chr{i}" for i in range(1, 21)]
    validation_chroms = ["chr21"]
    # test_chroms = ["chr22"]

    training_dataset = DNASeqDataset(
        "data/ENCFF896UZB.bed", "data/hg38.fa", training_chroms, allow_rc=True
    )
    validation_dataset = DNASeqDataset(
        "data/ENCFF896UZB.bed", "data/hg38.fa", validation_chroms, allow_rc=False
    )
    # test_dataset = DNASeqDataset("data/ENCFF896UZB.bed", "data/hg38.fa", test_chroms)

    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
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

    model: DNACNN = DNACNN().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0

        for sequences, labels in training_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # clear out gradients from previous epoch

            predictions, hidden_activation = model.forward_return_hidden(sequences)

            bceloss = criterion(predictions, labels)
            h_loss = (
                args.output_decay
                * torch.norm(hidden_activation, p=1)
                / sequences.size(0)
            )
            loss = bceloss + h_loss

            loss.backward()

            optimizer.step()

            apply_weight_norm_constraint(model, args.neuron_norm_max)

            running_loss += loss.item()
        avg_loss = nan
        if len(training_loader) != 0:
            avg_loss = running_loss / len(training_loader)

        model.eval()
        validation_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for validation_sequences, validation_labels in validation_loader:
                validation_sequences = validation_sequences.to(device)
                validation_labels = validation_labels.to(device)

                validation_logits = model(validation_sequences)

                batch_loss = criterion(validation_logits, validation_labels)
                validation_loss += batch_loss.item()

                probabilities = torch.sigmoid(validation_logits)
                validation_predictions = (probabilities >= 0.5).float()
                correct_predictions += (
                    (validation_predictions == validation_labels).sum().item()
                )
                total_predictions += validation_predictions.size(0)
        avg_validation_loss = validation_loss / len(validation_loader)
        validation_accuracy = correct_predictions / total_predictions

        scheduler.step(avg_validation_loss)

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

        logger.info(
            f"At epoch {epoch + 1} of {args.num_epochs} Training loss: {avg_loss:.4f} Validation loss: {avg_validation_loss:.4f} Validation accuracy: {validation_accuracy * 100:.4f}%"
        )
