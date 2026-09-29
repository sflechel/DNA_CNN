import json
import os
import numpy as np
import h5py
import logging
from pysam import FastaFile
import random
import bisect

from scripts.utils import get_gc_content, load_chrom_peaks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def build_exclusion_zones(
    peaks: list[tuple[int, int, int]], window_size: int
) -> list[tuple[int, int]]:

    all_zones: list[tuple[int, int]] = [
        (min(0, peak[0] - window_size), peak[1] + window_size) for peak in peaks
    ]

    merged: list[tuple[int, int]] = [all_zones[0]]
    for zone in all_zones[1:]:
        last = merged[-1]
        if zone[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], zone[1]))
        else:
            merged.append(zone)
    return merged


def is_excluded(start: int, end: int, zones: list[tuple[int, int]]) -> bool:
    zone_starts = [zone[0] for zone in zones]
    id_prev_zone = bisect.bisect_right(zone_starts, end) - 1
    if id_prev_zone >= 0 and zones[id_prev_zone][1] > start:
        return True
    return False


def find_and_pop_gc_match(candidate_gc: float, gcs: list[float]) -> float:
    return candidate_gc


def extract_negatives(
    chrom: str,
    gc_tolerance: float,
    window_size: int,
    fa: str,
    master_bed: str,
    target_ids: str,
    pos_stats: str,
    h5: str,
) -> None:
    with open(pos_stats, "r") as file:
        stats = json.load(file)
    with open(target_ids, "r") as file:
        ids = json.load(file)
    num_targets: int = len(ids["count"])

    nb_peaks: int = stats["count"]
    if nb_peaks <= 0:
        os.makedirs(os.path.dirname(h5), exist_ok=True)
        with h5py.File(h5, "w") as h5file:
            h5file.create_dataset("inputs", shape=(0, 4, window_size), dtype=np.float32)
            h5file.create_dataset("targets", shape=(0, num_targets), dtype=np.float32)
        logging.warning(f"No peaks found for {chrom}. Creating empty h5 dataset")
        exit(1)

    peaks = load_chrom_peaks(master_bed, chrom, ids["targets_to_ids"])
    # load chrom peaks return a sorted list
    exclusion_zones = build_exclusion_zones(peaks, window_size)

    with FastaFile(fa) as genome:
        if chrom not in genome.references:
            logging.error(f"Chromosome {chrom} not in genome")
            exit(1)
        chrom_len: int = genome.get_reference_length(chrom)

        pos_gcs: list[float] = sorted(stats["pos_gc_content"])
        attempts: int = 0
        max_attempts: int = 200

        while len(pos_gc) > 0 and attempts < max_attempts:
            attempts += 1
            seq_start = random.randint(0, chrom_len - window_size)
            seq_end = seq_start + window_size

            if is_excluded(seq_start, seq_end, exclusion_zones):
                continue

            seq_str: str = genome.fetch(chrom, seq_start, seq_end).upper()
            if "N" in seq_str:
                continue

            seq_gc: float = get_gc_content(seq_str)
            matched_gc: float = find_and_pop_gc_match(seq_gc, pos_gcs)


def main() -> None:
    extract_negatives(
        chrom=snakemake.wildcards.chrom,
        gc_tolerance=snakemake.config.gc_tolerance,
        window_size=snakemake.config.window_size,
        fa=snakemake.input.fa,
        master_bed=snakemake.input.master_bed,
        pos_stats=snakemake.input.pos_stats,
        target_ids=snakemake.input.target_ids,
        h5=snakemake.output.h5,
    )


if __name__ == "__main__":
    main()
