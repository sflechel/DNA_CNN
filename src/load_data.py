import pandas as pd
import pysam
import torch
from torch.utils.data import Dataset
from torch import Tensor
from dna_utils import one_hot_encode


class DNASeqDataset(Dataset):
    def __init__(self, bed_file: str, fasta_file: str, half_window=500):
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
        self.peaks = all_peaks[all_peaks["chrom"] == "chr22"]
        self.genome = None

    def __len__(self):
        return len(self.peaks)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        peak = self.peaks.iloc[index]
        if self.genome is None:
            self.genome = pysam.FastaFile(self.fasta_file)
        seq = load_seq_at_peak(
            genome, peak["chrom"], peak["start"], peak["end"], self.half_window
        )
        encoded = one_hot_encode(seq)
        return torch.tensor(encoded, dtype=torch.float32), torch.tensor(
            1.0, dtype=torch.float32
        )  # we also return the label, set to true


def load_seq_at_peak(
    genome: pysam.FastaFile, chr: str, start: int, peak: int, half_window: int
) -> str:
    newStart = start + peak - half_window
    end = newStart + half_window * 2
    if newStart < 0:
        start = 0
        end = 1000
    seq = genome.fetch(chr, newStart, end)
    if len(seq) < half_window * 2:
        seq = seq + seq.ljust(half_window * 2, "N")
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
