from math import nan
import torch
from torch.utils.data import DataLoader
from src.load_data import DNASeqDataset
from src.model import DNACNN
import torch.nn as nn
import torch.optim as optim
import logging
from pathlib import Path


def train(batch_size: int = 64, num_workers: int = 16):
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
        "data/ENCFF896UZB.bed", "data/hg38.fa", training_chroms
    )
    validation_dataset = DNASeqDataset(
        "data/ENCFF896UZB.bed", "data/hg38.fa", validation_chroms
    )
    # test_dataset = DNASeqDataset("data/ENCFF896UZB.bed", "data/hg38.fa", test_chroms)

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
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

    model = DNACNN().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    num_epochs = 10
    best_val_loss = float("inf")
    lambda2 = 1e-5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for sequences, labels in training_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # clear out gradients from previous epoch

            # features = model.feature_extractor(sequences.permute(0, 2, 1))
            # flattened = model.classifier[0](features)
            #
            # hidden_weights = model.classifier[1](flattened)
            # hidden_activation = model.classifier[2](hidden_weights)
            # hidden_dropout = model.classifier[3](hidden_activation)
            # predictions = model.classifier[4](hidden_dropout)

            # predictions = model(sequences)

            predictions, hidden_activation = model.forward_return_hidden(sequences)

            bceloss = criterion(predictions, labels)
            loss = bceloss + lambda2 * torch.norm(
                hidden_activation, p=1
            ) / sequences.size(0)

            loss.backward()

            optimizer.step()

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

        if avg_validation_loss < best_val_loss:
            best_val_loss = avg_validation_loss
            checkpoint_path: Path = Path("output/checkpoints/best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"New best model found. Saving at {checkpoint_path}")

        logger.info(
            f"At epoch {epoch + 1} of {num_epochs} Training loss: {avg_loss:.4f} Validation loss: {avg_validation_loss:.4f} Validation accuracy: {validation_accuracy * 100:.4f}%"
        )


if __name__ == "__main__":
    train()
