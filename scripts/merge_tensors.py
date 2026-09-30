import json
import os
import logging
from typing import cast
import h5py
import numpy as np
from numpy._typing import NDArray

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def warn_if_few_peaks(
    y_data: NDArray[np.float32],
    min_peak_warning: int,
    target_list: list[str],
    dataset_name: str,
) -> None:
    sum_per_target: NDArray[np.float32] = y_data.sum(axis=0)
    for name, count in zip(target_list, sum_per_target):
        if count < min_peak_warning:
            logging.warning(
                f"Only {count} peaks for target {name} in dataset {dataset_name}"
            )


def process_and_save_dataset(
    filepaths: list[str],
    output: str,
    targets: list[str],
    name: str,
    window_size: int,
    min_peak_warning: int,
) -> None:
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

    X_data: NDArray[np.float32] = np.zeros(
        [total_samples, 4, window_size], dtype=np.float32
    )
    y_data: NDArray[np.float32] = np.zeros(
        [total_samples, num_targets], dtype=np.float32
    )

    pos: int = 0
    for path in filepaths:
        with h5py.File(path, "r") as file:
            inputs = cast(h5py.Dataset, file["inputs"])
            num_samples: int = inputs.shape[0]
            if num_samples == 0:
                continue
            X_data[pos : pos + num_samples] = inputs[:]
            y_data[pos : pos + num_samples] = cast(h5py.Dataset, file["targets"])[:]
            pos += num_samples

    warn_if_few_peaks(y_data, min_peak_warning, targets, name)

    if total_samples > 0:
        permutation: NDArray[np.integer] = np.random.permutation(total_samples)
        X_data = X_data[permutation]
        y_data = y_data[permutation]

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with h5py.File(output, "w") as h5:
        h5.create_dataset("inputs", data=X_data, compression="gzip")
        h5.create_dataset("targets", data=y_data, compression="gzip")

        h5.create_dataset(
            "target_names",
            data=np.array(targets),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    logging.info(f"Wrote {total_samples} samples to {name} dataset at {output}")


def merge_tensors(
    val_chroms: list[str],
    test_chroms: list[str],
    window_size: int,
    min_peak_warning: int,
    target_ids: str,
    pos_h5s: list[str],
    neg_h5s: list[str],
    train_h5: str,
    val_h5: str,
    test_h5: str,
) -> None:
    with open(target_ids, "r") as file:
        targets: list[str] = json.load(file)["target_list"]

    all_h5s: list[str] = pos_h5s + neg_h5s

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
        min_peak_warning=min_peak_warning,
    )
    process_and_save_dataset(
        filepaths=split_files["test"],
        output=test_h5,
        targets=targets,
        name="test",
        window_size=window_size,
        min_peak_warning=min_peak_warning,
    )
    process_and_save_dataset(
        filepaths=split_files["train"],
        output=train_h5,
        targets=targets,
        name="train",
        window_size=window_size,
        min_peak_warning=min_peak_warning,
    )


def main() -> None:
    merge_tensors(
        val_chroms=snakemake.config["validation_chroms"],
        test_chroms=snakemake.config["test_chroms"],
        window_size=snakemake.config["window_size"],
        min_peak_warning=snakemake.config["min_peak_warning"],
        target_ids=snakemake.input.target_ids,
        pos_h5s=snakemake.input.pos_h5s,
        neg_h5s=snakemake.input.neg_h5s,
        train_h5=snakemake.output.train_h5,
        val_h5=snakemake.output.val_h5,
        test_h5=snakemake.output.test_h5,
    )


if __name__ == "__main__":
    main()
