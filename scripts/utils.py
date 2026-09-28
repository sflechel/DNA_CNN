from numpy._typing import NDArray
import numpy as np


def load_chrom_peaks(
    bed_path: str, target_chrom: str, targets_to_ids: dict[str, int]
) -> list[tuple[int, int, int]]:
    peaks = []
    with open(bed_path, "r") as file:
        for line in file:
            parts: list[str] = line.strip().split()
            if len(parts) < 4:
                continue
            chrom, start, end, target_name = (
                parts[0],
                int(parts[1]),
                int(parts[2]),
                parts[3],
            )
            if chrom == target_chrom and target_name in targets_to_ids:
                peaks.append((start, end, targets_to_ids[target_name]))

    peaks.sort(key=lambda x: x[0])
    return peaks


def get_gc_content(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq)


def one_hot_encode_sequences(seqs: list[str]) -> NDArray[np.float32]:
    num_seqs = len(seqs)
    seq_len = len(seqs[0])

    lookup = np.zeros([256, 4], dtype=np.float32)
    lookup[ord("A")] = [1, 0, 0, 0]
    lookup[ord("T")] = [0, 1, 0, 0]
    lookup[ord("C")] = [0, 0, 1, 0]
    lookup[ord("G")] = [0, 0, 0, 1]

    ascii_matrix = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8).reshape(
        num_seqs, seq_len
    )

    # 2. Array lookup produces (N, L, 4) -> transpose to (N, 4, L)
    return lookup[ascii_matrix].transpose(0, 2, 1)
