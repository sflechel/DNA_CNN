import torch
from torch.utils.data import Dataset
from torch import Tensor
import logging
import h5py
import numpy as np
from typing import cast


def reverse_complement(sequences: torch.Tensor) -> torch.Tensor:
    # assumes A=0, C=1, G=2, T=3
    flipped = torch.flip(sequences, dims=[0])
    rc = flipped[:, [3, 2, 1, 0]]
    return rc


class DNASeqDataset(Dataset):
    def __init__(
        self,
        h5_filepath: str,
        allow_rc: bool,
    ):
        self.allow_rc = allow_rc
        self.h5_path = h5_filepath
        with h5py.File(self.h5_path, "r") as file:
            inputs = file["inputs"]
            targets = file["targets"]
            assert isinstance(inputs, h5py.Dataset)
            assert isinstance(targets, h5py.Dataset)
            self.num_samples = inputs.shape[0]
            self.num_targets = targets.shape[1]
            self.window_size = inputs.shape[2]
            target_names = cast(h5py.Dataset, file["target_names"])
            self.target_names = list(target_names.asstr()[:])

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler("training_log.log"), logging.StreamHandler()],
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Number of peaks in dataset: {self.num_samples}")

    def __len__(self) -> int:
        return self.num_samples

    def _get_h5_handle(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
        return self.h5_file

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        h5file = self._get_h5_handle()
        inputs = h5file["inputs"]
        targets = h5file["targets"]
        assert isinstance(inputs, h5py.Dataset)
        assert isinstance(targets, h5py.Dataset)

        X = inputs[index].astype(np.float32)
        y = targets[index].astype(np.float32)

        if self.allow_rc and torch.rand(1).item() > 0.5:
            X = reverse_complement(X)

        return X, y

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()  # not strictly necessary, the garbage collector should take care of this
