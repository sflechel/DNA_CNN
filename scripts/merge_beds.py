import glob
import gzip
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def merge_beds(beds_dir: str, output_bed: str):
    os.makedirs(os.path.dirname(output_bed), exist_ok=True)

    bed_files = glob.glob(os.path.join(beds_dir, "*.bed.gz"))
    logging.info(f"Merging {len(bed_files)} peak files into '{output_bed}'...")

    total_peaks = 0

    with open(output_bed, "w") as out_f:
        for bed_path in sorted(bed_files):
            filename = os.path.basename(bed_path)
            # Filename format: TARGET_ACCESSION.bed.gz -> extract TARGET
            target_name = filename.split("_")[0]

            with gzip.open(bed_path, "rt") as in_f:
                for line in in_f:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue

                    chrom, start, end = parts[0], parts[1], parts[2]

                    if not (
                        chrom.startswith("chr")
                        and (chrom[3:].isdigit() or chrom[3:] in ["X", "Y"])
                    ):
                        logging.warning(
                            f"Nonstandard chromosome name {chrom}, dropping"
                        )
                        continue

                    out_f.write(f"{chrom}\t{start}\t{end}\t{target_name}\n")
                    total_peaks += 1

    logging.info(f"Consolidated {total_peaks} total peak records into '{output_bed}'.")


def main() -> None:
    merge_beds(
        beds_dir=snakemake.config["beds_dir"], output_bed=snakemake.output.master_bed
    )


if __name__ == "__main__":
    main()
