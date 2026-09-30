import json
import logging
import os
from numpy.typing import NDArray
import numpy as np
from pysam import FastaFile

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


def filter_peaks(
    min_peaks: int,
    peaks: list[tuple[str, int, int, NDArray[np.int32]]],
    target_list: list[str],
) -> tuple[list[tuple[str, int, int, list[int]]], list[str], dict[str, int]]:

    label_matrix: NDArray[np.int32] = np.vstack([p[3] for p in peaks])
    target_counts: NDArray[np.int32] = label_matrix.sum(axis=0)
    surviving_targets_mask: NDArray[np.bool_] = target_counts >= min_peaks

    for name, count, keep in zip(target_list, target_counts, surviving_targets_mask):
        logging.info(f"Target {name} has {count} peaks")
        if not keep:
            logging.info(f"Only {count} peaks for target {name}! Dropping it")

    surviving_target_list: list[str] = [
        target for target, keep in zip(target_list, surviving_targets_mask) if keep
    ]
    surviving_targets_to_ids: dict[str, int] = {
        t: i for i, t in enumerate(surviving_target_list)
    }

    filtered_label_matrix: NDArray[np.int32] = label_matrix[:, surviving_targets_mask]
    valid_peak_mask: NDArray[np.bool_] = filtered_label_matrix.sum(axis=1) > 0

    filtered_peaks: list[tuple[str, int, int, list[int]]] = []
    for keep, (chrom, start, end, _), new_label_vec in zip(
        valid_peak_mask, peaks, filtered_label_matrix
    ):
        if keep:
            active_labels: list[int] = np.flatnonzero(new_label_vec).tolist()
            filtered_peaks.append((chrom, start, end, active_labels))

    logging.info(
        f"Filtered peaks: {len(filtered_peaks)}/{len(peaks)} peaks retained "
        f"across {len(surviving_target_list)}/{len(target_list)} remaining targets."
    )

    return filtered_peaks, surviving_target_list, surviving_targets_to_ids


def merge_peaks(
    bed_path: str,
    fa: str,
    target_list: list[str],
    targets_to_ids: dict[str, int],
    window_size: int,
    inner_size: int,
) -> list[tuple[str, int, int, NDArray[np.int32]]]:

    peaks_per_chrom: dict[str, list[tuple[int, int, int]]] = {}
    chroms_seen: list[str] = []

    with open(bed_path, "r") as file:
        for line in file:
            parts: list[str] = line.strip().split()
            if len(parts) < 4:
                continue

            chrom = parts[0]
            if chrom not in chroms_seen:
                peaks_per_chrom[parts[0]] = []
                chroms_seen.append(parts[0])

            target: int = targets_to_ids[parts[3]]
            peaks_per_chrom[chrom].append((int(parts[1]), int(parts[2]), target))

    if not peaks_per_chrom:
        logging.error(f"No peaks found in {bed_path}!")
        exit(1)

    new_peaks: list[tuple[str, int, int, NDArray[np.int32]]] = []
    with FastaFile(fa) as genome:
        for chrom in peaks_per_chrom.keys():
            if chrom not in genome.references:
                logging.error(f"Chromosome {chrom} not in genome")
                exit(1)
            chrom_len: int = genome.get_reference_length(chrom)

            peaks = peaks_per_chrom[chrom]
            peaks.sort(key=lambda x: x[1])
            clusters = cluster_peaks(peaks, max_span=inner_size)

            for cluster in clusters:
                cluster_start = cluster[0][0]
                cluster_end = cluster[-1][1]

                center: int = (cluster_start + cluster_end) // 2
                new_peak_start: int = center - (window_size // 2)
                new_peak_end: int = center + (window_size // 2)

                if new_peak_start < 0 or new_peak_end > chrom_len:
                    continue

                peak_targets: NDArray[np.integer] = np.zeros(
                    [len(target_list)], dtype=np.int32
                )
                for peak in cluster:
                    peak_targets[peak[2]] = 1

                new_peaks.append((chrom, new_peak_start, new_peak_end, peak_targets))

    return new_peaks


def map_targets_to_ids(bed_path: str) -> tuple[list[str], dict[str, int]]:
    targets: set[str] = set()
    with open(bed_path, "r") as file:
        for line in file:
            parts: list[str] = line.strip().split()
            if len(parts) < 4:
                continue
            targets.add(parts[3])

    target_list: list[str] = sorted(list(targets))
    targets_to_ids: dict[str, int] = {name: id for id, name in enumerate(target_list)}
    logging.info(f"Found {len(target_list)} unique targets")

    return target_list, targets_to_ids


def write_targets(
    merged_peaks: list[tuple[str, int, int, list[int]]],
    target_list: list[str],
    targets_to_ids: dict[str, int],
    targets_path: str,
    bed_path: str,
) -> None:
    os.makedirs(os.path.dirname(targets_path), exist_ok=True)
    with open(targets_path, "w") as file:
        json.dump({"target_list": target_list, "targets_to_ids": targets_to_ids}, file)
    logging.info(f"Wrote target list and target ids to {targets_path}")

    total_peaks: int = 0
    os.makedirs(os.path.dirname(bed_path), exist_ok=True)
    with open(bed_path, "w") as file:
        for peak in merged_peaks:
            chrom, start, end = peak[0], peak[1], peak[2]

            target_ids_str = "_".join(map(str, peak[3]))

            file.write(f"{chrom}\t{start}\t{end}\t{target_ids_str}\n")
            total_peaks += 1

    logging.info(f"{total_peaks} total peaks remain, written to {bed_path}")


def main() -> None:
    bed_path: str = snakemake.input.master_bed
    target_list, targets_to_ids = map_targets_to_ids(bed_path=bed_path)

    merged_peaks = merge_peaks(
        bed_path=bed_path,
        target_list=target_list,
        targets_to_ids=targets_to_ids,
        window_size=snakemake.config["window_size"],
        inner_size=snakemake.config["inner_size"],
        fa=snakemake.input.fa,
    )

    filtered_peaks, filtered_list, filtered_ids = filter_peaks(
        min_peaks=snakemake.config["min_peaks_filter"],
        peaks=merged_peaks,
        target_list=target_list,
    )

    write_targets(
        merged_peaks=filtered_peaks,
        target_list=filtered_list,
        targets_to_ids=filtered_ids,
        targets_path=snakemake.output.target_ids,
        bed_path=snakemake.output.filtered_master_bed,
    )


if __name__ == "__main__":
    main()

# to check for tomfoolery:
# grep '_' data/processed/filtered_master.bed | head -n 1
## then, replacing with the coordinates you got:
# grep -P "^chr1\t" data/processed/master.bed | awk '$2 >= 1000000 && $3 <= 1005000'
