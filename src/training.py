from torch.utils.data import DataLoader
from load_data import DNASeqDataset


def load_and_train(batch_size: int, num_workers: int):
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

    print(f"Sequence batch shape: {sequence.shape}")
    print(f"Labels batch shape: {labels.shape}")
    print(f"Labels mean: {labels.mean().item()}")
