import json
import os
import logging
from typing import cast
import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def process_and_save_dataset(
    filepaths: list[str],
    output: str,
    targets: list[str],
    name: str,
    window_size: int,
) -> None:
    """Get per-chromosome positive and negative sequences, shuffle them and output them as a h5 dataset"""
    num_targets: int = len(targets)
    total_samples: int = 0

    for path in filepaths:
        with h5py.File(path, "r") as file:
            num_samples: int = cast(h5py.Dataset, file["inputs"]).shape[0]
            total_samples += num_samples

    if total_samples == 0:
        logging.warning(f"Dataset {name} has 0 samples!")
    else:
        logging.info(f"Dataset {name} has {total_samples} samples")

    # out-of-core shuffling
    tmp_name: str = f"{output}_tmp"
    with h5py.File(tmp_name, "w") as outfile:
        sequences = outfile.create_dataset(
            "inputs",
            shape=(total_samples, 4, window_size),
            dtype="uint8",
            chunks=(256, 4, window_size),
            compression="lzf",
        )
        labels = outfile.create_dataset(
            "targets",
            shape=(total_samples, num_targets),
            dtype="float32",
            chunks=(256, num_targets),
            compression="lzf",
        )

        pos = 0
        for path in filepaths:
            with h5py.File(path, "r") as infile:
                inputs = cast(h5py.Dataset, infile["inputs"])
                trgts = cast(h5py.Dataset, infile["targets"])
                num_samples: int = inputs.shape[0]
                sequences[pos : pos + num_samples] = inputs[:]
                labels[pos : pos + num_samples] = trgts[:]
                pos += num_samples

    perm = np.random.permutation(total_samples)
    chunk_size = 100000000  # adjust based on available RAM

    with h5py.File(tmp_name, "r") as src, h5py.File(output, "w") as dst:
        dst_inputs = dst.create_dataset(
            "inputs",
            shape=(total_samples, 4, window_size),
            dtype="uint8",
            chunks=(256, 4, window_size),
            compression="lzf",
        )
        dst_targets = dst.create_dataset(
            "targets",
            shape=(total_samples, num_targets),
            dtype="float32",
            chunks=(256, num_targets),
            compression="lzf",
        )
        dst.create_dataset("target_names", data=targets)

        for i in range(0, total_samples, chunk_size):
            chunk_perm = perm[i : i + chunk_size]  # sort for sequential HDF5 reads
            sort_order = np.argsort(chunk_perm)
            sorted_idx = chunk_perm[sort_order]

            tmp_inputs = src["inputs"][sorted_idx]  # type: ignore[index]
            tmp_targets = src["targets"][sorted_idx]  # type: ignore[index]

            unsorted = np.argsort(sort_order)
            dst_inputs[i : i + chunk_size] = tmp_inputs[unsorted]  # type: ignore[index]
            dst_targets[i : i + chunk_size] = tmp_targets[unsorted]  # type: ignore[index]

    os.remove(tmp_name)
    logging.info(f"Wrote {total_samples} samples to {name} dataset at {output}")


def merge_tensors(
    val_chroms: list[str],
    test_chroms: list[str],
    window_size: int,
    target_ids: str,
    pos_h5s: list[str],
    neg_h5s: list[str],
    test_h5s: list[str],
    train_h5: str,
    val_h5: str,
    test_h5: str,
) -> None:
    """Get all chromosome bed files, split them into training, validation and test datasets"""
    with open(target_ids, "r") as file:
        targets: list[str] = json.load(file)["target_list"]

    all_h5s: list[str] = pos_h5s + neg_h5s + test_h5s

    split_files: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for h5_path in all_h5s:
        filename = os.path.basename(h5_path)
        chrom = filename.split("_")[0]

        if chrom in val_chroms:
            split_files["val"].append(h5_path)
        elif chrom in test_chroms:
            split_files["test"].append(h5_path)
        else:
            split_files["train"].append(h5_path)

    process_and_save_dataset(
        filepaths=split_files["val"],
        output=val_h5,
        targets=targets,
        name="val",
        window_size=window_size,
    )
    process_and_save_dataset(
        filepaths=split_files["test"],
        output=test_h5,
        targets=targets,
        name="test",
        window_size=window_size,
    )
    process_and_save_dataset(
        filepaths=split_files["train"],
        output=train_h5,
        targets=targets,
        name="train",
        window_size=window_size,
    )


def main() -> None:
    merge_tensors(
        val_chroms=snakemake.config["validation_chroms"],
        test_chroms=snakemake.config["test_chroms"],
        window_size=snakemake.config["window_size"],
        target_ids=snakemake.input.target_ids,
        pos_h5s=snakemake.input.pos_h5s,
        neg_h5s=snakemake.input.neg_h5s,
        test_h5s=snakemake.input.test_h5s,
        train_h5=snakemake.output.train_h5,
        val_h5=snakemake.output.val_h5,
        test_h5=snakemake.output.test_h5,
    )


if __name__ == "__main__":
    main()
