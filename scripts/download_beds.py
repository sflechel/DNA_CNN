import gzip
import os
import sys
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def create_robust_session():
    """Exponential falloff retry rules for API querying"""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=3,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def count_gzipped_lines(filepath):
    count = 0
    with gzip.open(filepath, "rt") as f:
        for _ in f:
            count += 1
    return count


def main():
    """Query ENCODE for ChIPseq data matching configured parameters"""
    config = snakemake.config
    outdir = config["beds_outdir"]
    os.makedirs(outdir, exist_ok=True)

    session = create_robust_session()
    headers = {
        "accept": "application/json",
        "User-Agent": "GenomicsPipeline/1.0 (Academic Research)",
    }

    logging.info(f"Querying ENCODE API for {config['cell_type']}...")

    search_url = "https://www.encodeproject.org/search/"
    params = [
        ("type", "Experiment"),
        ("assay_term_name", config["assay_type"]),
        ("biosample_ontology.term_name", config["cell_type"]),
        ("status", "released"),
        ("format", "json"),
        ("limit", "all"),
        ("field", "@id"),
        ("field", "target"),
        ("field", "files"),
        ("field", "audit"),
    ]

    response = session.get(search_url, params=params, headers=headers, timeout=60)
    if response.status_code != 200:
        logging.error(f"API request failed (Status: {response.status_code})")
        sys.exit(1)

    experiments = response.json().get("@graph", [])
    logging.info(f"Successfully retrieved metadata for {len(experiments)} experiments")

    success_count = 0

    for exp in experiments:
        if "ERROR" in exp.get("audit", {}):
            continue

        target_obj = exp.get("target", {})
        if not target_obj:
            continue

        target_name = target_obj.get("label")
        if not target_name:
            continue

        for f in exp.get("files", []):
            if (
                f.get("file_format") == "bed"
                and f.get("file_format_type") == "narrowPeak"
                and f.get("output_type") == "IDR thresholded peaks"
                and f.get("assembly") == config["assembly"]
                and f.get("status") == "released"
            ):
                bed_file_url = f"https://www.encodeproject.org{f['href']}"
                file_acc = f["accession"]

                dest_path = os.path.join(outdir, f"{target_name}_{file_acc}.bed.gz")

                if os.path.exists(dest_path):
                    success_count += 1
                    break

                logging.info(f"Downloading {target_name} ({file_acc}) -> {dest_path}")
                file_res = session.get(bed_file_url, stream=True, timeout=60)

                with open(dest_path, "wb") as out_f:
                    for chunk in file_res.iter_content(chunk_size=8192):
                        out_f.write(chunk)

                peak_count = count_gzipped_lines(dest_path)
                if peak_count < config["min_peaks"]:
                    logging.info(
                        f"{target_name} ({file_acc}) has only {peak_count} peaks (Threshold: {config['min_peaks']}). Deleting."
                    )
                    os.remove(dest_path)
                    continue

                logging.info(f"{target_name} ({file_acc}) has {peak_count} peaks.")
                success_count += 1
                time.sleep(0.1)
                break

    logging.info(f"Downloaded {success_count} valid peak files into '{outdir}'.")

    with open(snakemake.output.sentinel, "w") as f:
        f.write("done")


if __name__ == "__main__":
    main()
