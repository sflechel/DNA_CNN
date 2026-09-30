import json
import logging
import os

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


def merge_peaks(
    bed_path: str,
    target_list: list[str],
    targets_to_ids: dict[str, int],
    window_size: int,
) -> tuple[list[str], dict[str, int]]:

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

    for chrom in peaks_per_chrom.keys():
        peaks = peaks_per_chrom[chrom]
        clusters = cluster_peaks(peaks, max_span=window_size)

    return target_list, targets_to_ids


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
    target_list: list[str], targets_to_ids: dict[str, int], output_path: str
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as file:
        json.dump({"target_list": target_list, "targets_to_ids": targets_to_ids}, file)
    logging.info(f"Wrote target list and target ids to {output_path}")


def main() -> None:
    bed_path: str = snakemake.input.master_bed
    target_list: list[str]
    targets_to_ids: dict[str, int]
    target_list, targets_to_ids = map_targets_to_ids(bed_path=bed_path)

    filtered_list: list[str]
    filtered_ids: dict[str, int]
    filtered_list, filtered_ids = merge_peaks(
        bed_path=bed_path,
        target_list=target_list,
        targets_to_ids=targets_to_ids,
        window_size=snakemake.config["window_size"],
    )

    write_targets(
        target_list=filtered_list,
        targets_to_ids=filtered_ids,
        output_path=skamemake.output.target_ids,
    )


if __name__ == "__main__":
    main()
