import argparse
import pandas as pd
import pysam
import random
import itertools
import numpy as np
from src.load_data import load_seq_at_peak
import pathlib


def main():
    args = get_arguments()
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
    with pysam.FastaFile(args.fasta_file) as genome:
        peaks = pd.read_csv(args.bed_file, sep="\t", header=None, names=cols)
        peaks_per_chrom: dict[str, pd.DataFrame] = {
            str(name): group for name, group in peaks.groupby("chrom")
        }
        all_offpeaks = []
        for chrom in peaks_per_chrom:
            if chrom not in genome.references:
                print(f"Skipping {chrom}: not in reference genome")
                continue
            chrom_len = genome.get_reference_length(chrom)
            safe_zones = get_safe_zones(
                peaks_per_chrom[chrom],
                chrom_len,
                args.half_window,
            )
            lengths = [zone[1] - zone[0] for zone in safe_zones]
            if not lengths:
                continue
            chrom_offpeaks = generate_offpeaks(
                safe_zones,
                lengths,
                chrom,
                peaks_per_chrom[chrom],
                args.half_window,
                genome,
            )
            all_offpeaks.extend(chrom_offpeaks)
        offpeaks_df: pd.DataFrame = pd.DataFrame(all_offpeaks)
        offpeaks_df = offpeaks_df.sort_values(
            by=["chrom", "start"]
        )  # BED files are sorted by chromosome and starting pos
        offpeaks_df = offpeaks_df[
            cols
        ]  # force column order to follow the standard format
        input_path = pathlib.Path(args.bed_file)
        offpeak_filename = input_path.parent._str + "/off_" + input_path.name
        offpeaks_df.to_csv(offpeak_filename, sep="\t", header=False, index=False)
        print(f"Successfully wrote {len(offpeaks_df)} negatives to {offpeak_filename}")


def get_arguments() -> argparse.Namespace:
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
    return parser.parse_args()


def generate_offpeaks(
    safe_zones,
    lengths: list[int],
    chrom: str,
    peaks: pd.DataFrame,
    half_window: int,
    genome: pysam.FastaFile,
):
    thresholds = list(itertools.accumulate(lengths))
    peaks_count = len(peaks)
    offpeak_positions = []
    max_attempts = peaks_count * 100
    attempts = 0
    window = half_window * 2
    while len(offpeak_positions) < peaks_count and attempts < max_attempts:
        attempts += 1
        zone = random.choices(safe_zones, cum_weights=thresholds, k=1)[0]
        start = random.randint(zone[0], zone[1] - window)
        seq = genome.fetch(chrom, start, start + window).upper()
        target_gc, std_dev = compute_average_gc_of_peaks(peaks, genome, half_window)
        if "N" in seq:
            continue
        gc_content = seq.count("G") + seq.count("C")
        if abs(gc_content - target_gc) <= std_dev:
            write_offpeak(offpeak_positions, chrom, start, window)
    if attempts >= max_attempts:
        print(
            f"Warning: {chrom} hit max attempts ({max_attempts}). Only found {len(offpeak_positions)}/{peaks_count} negatives."
        )
        print(
            "Your GC constraints might be too tight for this chromosome's composition."
        )
    return offpeak_positions


def write_offpeak(offpeaks, chrom: str, start: int, window: int):
    offpeaks.append(
        {
            "chrom": chrom,
            "start": start,
            "end": start + window,
            "name": ".",
            "score": "0",
            "strand": "+",
            "sig": "0",
            "p": "0",
            "q": "0",
            "peak": window / 2,  # center of the negative peak
        }
    )


def compute_average_gc_of_peaks(
    peaks: pd.DataFrame, genome: pysam.FastaFile, half_window: int
) -> tuple[float, float]:
    gcs: list[int] = []
    for peak in peaks.itertuples(index=False):
        seq = load_seq_at_peak(
            genome, str(peak.chrom), peak[1], peak[9], half_window
        )  # peak[1] is start, peak[9] is peak.
        gc = seq.count("G") + seq.count("C")
        gcs.append(gc)
    return float(np.mean(gcs)), float(np.std(gcs))


def get_safe_zones(peaks: pd.DataFrame, chrom_len: int, half_window: int):
    window = half_window * 2
    intervals = peaks[["start", "end"]].sort_values(by="start").values.tolist()
    merged = []
    for start, end in intervals:
        if not merged:
            merged.append([start, end])
        else:
            _, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1][1] = max(prev_end, end)
            else:
                merged.append([start, end])

    safe_zones = []
    cursor = 0
    for start, end in merged:
        if start - cursor >= window:
            safe_zones.append((cursor, start))
        cursor = end

    # Check the tail end of the chromosome
    if chrom_len - cursor >= window:
        safe_zones.append((cursor, chrom_len))

    return safe_zones


if __name__ == "__main__":
    main()
