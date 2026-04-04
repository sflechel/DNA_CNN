import argparse
import pandas as pd
import pysam
from load_data import load_seq_at_peak


def main():
    parser = argparse.ArgumentParser(
        description="Generate annotations for bed files (GC-content)"
    )
    parser.add_argument(
        "bed_file", type=str, help="Path to the input .bed file (positives)"
    )
    parser.add_argument(
        "fasta_file", type=str, help="Path to the input genome .fasta file"
    )
    parser.add_argument(
        "--half_window", type=int, default=500, help="Half window size (default: 500)"
    )
    args = parser.parse_args()
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
    peaks = pd.read_csv(args.bed_file, sep="\t", header=None, names=cols)
    off_peaks_file = open("off" + args.bed_file, "w")
    with pysam.FastaFile(args.fasta_file) as genome:
        gc_total = 0
        i = 0
        for _, row in peaks.itertuples():
            seq = load_seq_at_peak(
                genome, row.chrom, row.start, row.peak, args.half_window
            ).upper()
            gc_total += seq.count("G")
            gc_total += seq.count("C")


if __name__ == "__main__":
    main()
