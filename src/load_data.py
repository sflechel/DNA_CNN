import pandas as pd
import pysam
import torch
from torch.utils.data import Dataset
from torch import Tensor
from src.dna_utils import one_hot_encode
import pathlib


class DNASeqDataset(Dataset):
    def __init__(
        self, bed_file: str, fasta_file: str, chromosoms: list[str], half_window=500
    ):
        self.half_window = half_window
        self.fasta_file = fasta_file
        cols = [
            "chrom",
            "start",
            "end",
            "name",
            "score",
            "strand",
            "sig",
            "p",
            "q",
            "peak",
        ]
        all_peaks = pd.read_csv(bed_file, sep="\t", names=cols)
        all_peaks["label"] = 1.0
        peak_path = pathlib.Path(bed_file)
        offpeak_filename = peak_path.parent / (f"/off_{peak_path.name}")
        all_offpeaks = pd.read_csv(offpeak_filename, sep="\t", names=cols)
        all_offpeaks["label"] = 0.0

        all_data = pd.concat([all_peaks, all_offpeaks], axis=0).reset_index(drop=True)
        self.peaks = all_data[all_data["chrom"].isin(chromosoms)].reset_index(drop=True)
        self.genome = None

    def __len__(self) -> int:
        return len(self.peaks)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        peak = self.peaks.iloc[index]
        if (
            self.genome is None
        ):  # each object opens its own FD, the first time __getitem__ is called
            self.genome = pysam.FastaFile(self.fasta_file)
        seq = load_seq_at_peak(
            self.genome, peak["chrom"], peak["start"], peak["end"], self.half_window
        )
        encoded = one_hot_encode(seq)
        return torch.tensor(encoded, dtype=torch.float32), torch.tensor(
            peak["label"], dtype=torch.float32
        )  # we also return the label

    def __del__(self):
        if self.genome is not None:
            self.genome.close()  # not strictly necessary, the garbage collector should take care of this


def load_seq_at_peak(
    genome: pysam.FastaFile, chr: str, start: int, peak: int, half_window: int
) -> str:
    newStart: int = start + peak - half_window
    end = newStart + half_window * 2
    if newStart < 0:
        newStart = 0
        end = 1000
    seq = genome.fetch(chr, newStart, end)
    if len(seq) < half_window * 2:
        seq = seq.ljust(half_window * 2, "N")
    return seq


if __name__ == "__main__":
    NARROW_PEAK_COLUMNS = [
        "chrom",
        "start",
        "end",
        "name",
        "score",
        "strand",
        "signalValue",
        "p",
        "q",
        "peak",
    ]
    peaks = pd.read_csv(
        "data/ENCFF896UZB.bed", sep="\t", header=None, names=NARROW_PEAK_COLUMNS
    )
    print(peaks.keys())
    chr22peaks = peaks[peaks["chrom"] == "chr22"]
    # chr22peaks = peaks.query("0==chr22")
    testPeak = chr22peaks.iloc[0]
    print(chr22peaks.iloc[0])
    with pysam.FastaFile("data/chr22.fa") as genome:  # will close FD when read over
        seq = load_seq_at_peak(
            genome, "chr22", testPeak["start"], testPeak["peak"], 500
        )
    if seq:
        print(seq)
        encoded = one_hot_encode(seq)
        print(encoded)
