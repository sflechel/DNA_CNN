import json
import logging
import os
from numpy._typing import NDArray
from pysam import FastaFile
import numpy as np
import h5py

from scripts.utils import get_gc_content, load_chrom_peaks, one_hot_encode_sequences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def cluster_peaks(
    peaks: list[tuple[int, int, int]], max_span: int
) -> list[list[tuple[int, int, int]]]:
    if not peaks:
        return []
    clusters = []
    current_cluster = [peaks[0]]

    for p in peaks[1:]:
        curr_cluster_start = current_cluster[0][0]
        if p[1] <= curr_cluster_start + max_span:
            current_cluster.append(p)
        else:
            clusters.append(current_cluster)
            current_cluster = [p]

    if current_cluster:
        clusters.append(current_cluster)

    return clusters


def extract_positives(
    chrom: str,
    window_size: int,
    inner_size: int,
    target_ids: str,
    master_bed: str,
    fa: str,
    stats: str,
    h5: str,
) -> None:
    seqs: list[str] = []
    labels: list[NDArray[np.float32]] = []
    peak_gc_contents: list[float] = []

    with open(target_ids, "r") as file:
        ids = json.load(file)
    num_targets: int = len(ids["target_list"])

    peaks = load_chrom_peaks(master_bed, chrom, ids["targets_to_ids"])
    # load_chrom_peaks return a list sorted by peak starts
    if not peaks:
        logging.error(f"Found no peaks for chromosome {chrom}")

    clusters = cluster_peaks(peaks, max_span=inner_size)

    with FastaFile(fa) as genome:
        if chrom not in genome.references:
            logging.error(f"Chromosome {chrom} not in genome")
            exit(1)
        chrom_len: int = genome.get_reference_length(chrom)

        for cluster in clusters:
            cluster_start = cluster[0][0]
            cluster_end = cluster[-1][1]

            center: int = (cluster_start + cluster_end) // 2
            seq_start: int = center - (window_size // 2)
            seq_end: int = center + (window_size // 2)

            if seq_start < 0 or seq_end > chrom_len:
                continue

            seq_str: str = genome.fetch(chrom, seq_start, seq_end).upper()
            if len(seq_str) != window_size or "N" in seq_str:
                logging.info("Peak sequence contains N, dropping")
                continue

            label: NDArray[np.float32] = np.zeros(num_targets, dtype=np.float32)
            for peak in cluster:
                label[peak[2]] = 1.0

            seqs.append(seq_str)
            labels.append(label)
            peak_gc_contents.append(get_gc_content(seq_str))

    logging.info(f"Found {len(seqs)} true peaks in chromosome {chrom}")

    os.makedirs(os.path.dirname(stats), exist_ok=True)
    with open(stats, "w") as file:
        json.dump(
            {"count": len(peak_gc_contents), "pos_gc_content": peak_gc_contents}, file
        )

    os.makedirs(os.path.dirname(h5), exist_ok=True)
    with h5py.File(h5, "w") as h5file:
        if seqs:
            h5file.create_dataset(
                "inputs", data=one_hot_encode_sequences(seqs), compression="gzip"
            )
            h5file.create_dataset(
                "targets", data=np.array(labels, dtype=np.float32), compression="gzip"
            )
        else:
            h5file.create_dataset("inputs", shape=(0, 4, window_size), dtype=np.float32)
            h5file.create_dataset("targets", shape=(0, num_targets), dtype=np.float32)


def main() -> None:
    chrom: str = snakemake.wildcards.chrom
    window_size: int = snakemake.config["window_size"]
    inner_size: int = snakemake.config["inner_size"]

    target_ids: str = snakemake.input.target_ids
    master_bed: str = snakemake.input.master_bed
    fa: str = snakemake.input.fa

    stats: str = snakemake.output.stats
    h5: str = snakemake.output.h5

    extract_positives(
        chrom=chrom,
        window_size=window_size,
        inner_size=inner_size,
        target_ids=target_ids,
        master_bed=master_bed,
        fa=fa,
        stats=stats,
        h5=h5,
    )


if __name__ == "__main__":
    main()
