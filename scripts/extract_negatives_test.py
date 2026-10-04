import numpy as np
import pybedtools
import pysam
import os
import h5py
from scripts.utils import one_hot_encode_sequences
import json
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def build_test_dataset(
    chrom: str,
    window_size: int,
    master_bed_path: str,
    fasta_path: str,
    target_ids: str,
    output_path: str,
):
    with pysam.FastaFile(fasta_path) as fasta:
        chrom_len: int = fasta.get_reference_length(chrom)

        master_bed = pybedtools.BedTool(master_bed_path)
        chrom_positives = master_bed.filter(lambda x: x.chrom == chrom).sort()

        logging.info(
            f"Found {len(chrom_positives):,} positive peak regions on {chrom} in master bed"
        )

        # 3. Extract Negatives: Take complement of master.bed on chr22
        # Define chr22 genome bounds for complement operation
        chrom_genome = pybedtools.BedTool(
            f"{chrom}\t0\t" + str(chrom_len), from_string=True
        )
        complements = chrom_genome.subtract(chrom_positives)

        # Tile non-peak regions into contiguous 1,000-bp windows
        neg_bins = pybedtools.BedTool().window_maker(
            b=complements, w=window_size, s=window_size
        )

        seqs: list[str] = []
        for b in neg_bins:
            if (b.end - b.start) < window_size:
                continue

            seq = fasta.fetch(chrom, b.start, b.end)
            if "N" in seq or "n" in seq:
                continue

            seqs.append(seq)
        nb_samples = len(seqs)
        logging.info(
            f"Extracted {nb_samples:,} offpeaks in chromosome {chrom} for test dataset"
        )

    with open(target_ids, "r") as file:
        ids = json.load(file)
    num_targets: int = len(ids["target_list"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as h5file:
        h5file.create_dataset(
            "inputs", data=one_hot_encode_sequences(seqs), compression="gzip"
        )
        h5file.create_dataset(
            "targets",
            data=np.zeros([len(seqs), num_targets], dtype=np.float32),
            compression="gzip",
        )
        logging.info(f"Offpeaks exported to {output_path}")


def main() -> None:
    build_test_dataset(
        chrom=snakemake.wildcards.chrom,
        window_size=snakemake.config["window_size"],
        master_bed_path=snakemake.input.master_bed,
        fasta_path=snakemake.input.fa,
        target_ids=snakemake.input.target_ids,
        output_path=snakemake.output.h5,
    )


if __name__ == "__main__":
    main()
