import pandas as pd
import pysam

from dna_utils import one_hot_encode


def load_seq_at_peak(
    genome: pysam.FastaFile, chr: str, start: int, peak: int, half_window: int
) -> str:
    newStart = start + peak - half_window
    end = newStart + half_window * 2
    if newStart < 0:
        start = 0
        end = 1000
    return genome.fetch(chr, newStart, end)


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
