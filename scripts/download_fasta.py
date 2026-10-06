import os
import gzip
import shutil
import logging
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def download_fasta():
    """Download whole genome FASTA file"""
    config = snakemake.config
    output_dir = config["fasta_outdir"]
    fasta = config["fasta"]
    fasta_url = config["fasta_url"]
    os.makedirs(output_dir, exist_ok=True)
    gz_path = os.path.join(output_dir, f"{fasta}.gz")
    fa_path = os.path.join(output_dir, fasta)

    if os.path.exists(fa_path):
        logging.info(f"Reference FASTA already exists at {fa_path}")
        return

    logging.info("Fetching hg38 reference genome from UCSC...")
    response = requests.get(fasta_url, stream=True)
    response.raise_for_status()

    with open(gz_path, "wb") as f_out:
        for chunk in response.iter_content(chunk_size=8192):
            f_out.write(chunk)

    logging.info(f"Unzipping {gz_path}...")
    with gzip.open(gz_path, "rb") as f_in, open(fa_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    os.remove(gz_path)
    logging.info(f"Reference FASTA ready at {fa_path}")


if __name__ == "__main__":
    download_fasta()
