from math import nan
import torch
from torch.utils.data import DataLoader
from load_data import DNASeqDataset
from model import DNACNN
import torch.nn as nn
import torch.optim as optim
import logging
import wandb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("training_log.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

wandb.init(
    project="DNA_CNN",
    config={
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 64,
        "architecture": "CNN-1D",
    },
)


def train(batch_size: int, num_workers: int):
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

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for sequences, labels in training_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # clear out gradients from previous epoch

            predictions = model(sequences)

            loss = criterion(predictions, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            wandb.log({"batch_loss:": loss.item()})
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

                batch_loss = criterion(validation_sequences, validation_labels)
                validation_loss += batch_loss.item()

                probabilities = torch.sigmoid(validation_logits)
                validation_predictions = (probabilities >= 0.5).float()
                correct_predictions += (
                    (validation_predictions == validation_labels).sum().item()
                )
                total_predictions += validation_predictions.size(0)
        avg_validation_loss = validation_loss / len(validation_loader)
        validation_accuracy = correct_predictions / total_predictions

        wandb.log(
            {
                "epoch": epoch + 1,
                "training_loss": avg_loss,
                "validation_loss": avg_validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        print(
            f"At epoch {epoch + 1} of {num_epochs} Training loss: {avg_loss:.4f} Validation loss: {avg_validation_loss:.4f} Validation accuracy: {validation_accuracy * 100:.4f}%"
        )
