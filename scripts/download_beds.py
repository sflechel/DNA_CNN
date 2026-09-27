import gzip
import os
import sys
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests


def create_robust_session():
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
    try:
        with gzip.open(filepath, "rt") as f:
            for _ in f:
                count += 1
    except (EOFError, gzip.BadGzipFile):
        return 0
    return count


def main():
    outdir = snakemake.config["beds_outdir"]
    os.makedirs(outdir, exist_ok=True)

    assembly_query = (
        "GRCh38"
        if snakemake.config["assembly"] == "hg38"
        else snakemake.config["assembly"]
    )

    session = create_robust_session()
    headers = {
        "accept": "application/json",
        "User-Agent": "GenomicsPipeline/1.0 (Academic Research)",
    }

    print("[SEARCHING] Fetching list of human targets from ENCODE...")
    targets_url = "https://www.encodeproject.org/search/"
    target_params = [
        ("type", "Target"),
        ("organism.scientific_name", "Homo sapiens"),
        ("format", "json"),
        ("limit", "all"),
        ("field", "name"),
        ("field", "label"),
    ]

    response = session.get(
        targets_url, params=target_params, headers=headers, timeout=60
    )
    if response.status_code != 200:
        print(f"[ERROR] Failed to fetch targets list (Status: {response.status_code})")
        sys.exit(1)

    targets = response.json().get("@graph", [])
    print(
        f"[INFO] Found {len(targets)} human targets. Querying {snakemake.config['cell_type']} ChIP-seq per target..."
    )

    success_count = 0
    downloaded_targets = set()

    for t in targets:
        target_label = t.get("label") or t.get("name")
        target_internal_name = t.get("name")
        if not target_label or not target_internal_name:
            continue

        clean_target = target_label.split("-")[0]
        if clean_target in downloaded_targets:
            continue

        exp_search_url = "https://www.encodeproject.org/search/"
        exp_params = [
            ("type", "Experiment"),
            ("assay_term_name", snakemake.config["assay_type"]),
            ("biosample_ontology.term_name", snakemake.config["cell_type"]),
            ("target.name", f"/targets/{target_internal_name}/"),
            ("status", "released"),
            ("format", "json"),
            ("limit", "5"),
            ("field", "@id"),
            ("field", "files"),
            ("field", "audit"),
        ]

        exp_res = session.get(
            exp_search_url, params=exp_params, headers=headers, timeout=30
        )
        if exp_res.status_code != 200:
            continue

        experiments = exp_res.json().get("@graph", [])
        if not experiments:
            continue

        found_for_target = False
        for exp in experiments:
            if found_for_target:
                break
            if "ERROR" in exp.get("audit", {}):
                continue

            for f in exp.get("files", []):
                if (
                    f.get("file_format") == "bed"
                    and f.get("output_type") == "optimal idr thresholded peaks"
                    and f.get("assembly") == assembly_query
                    and f.get("status") == "released"
                ):
                    bed_file_url = f"https://www.encodeproject.org{f['href']}"
                    file_acc = f["accession"]

                    dest_path = os.path.join(
                        outdir, f"{clean_target}_{file_acc}.bed.gz"
                    )

                    if os.path.exists(dest_path):
                        peak_count = count_gzipped_lines(dest_path)
                        if peak_count >= snakemake.config["min_peaks"]:
                            print(
                                f"[INFO] {clean_target} already secured ({file_acc}). Skipping."
                            )
                            success_count += 1
                            downloaded_targets.add(clean_target)
                            found_for_target = True
                            break
                        else:
                            os.remove(dest_path)

                    print(f"[DOWNLOADING] {clean_target} ({file_acc}) -> {dest_path}")
                    try:
                        file_res = session.get(bed_file_url, stream=True, timeout=60)
                        file_res.raise_for_status()
                        with open(dest_path, "wb") as out_f:
                            for chunk in file_res.iter_content(chunk_size=8192):
                                out_f.write(chunk)
                    except requests.exceptions.RequestException as e:
                        print(
                            f"[WARNING] Network error downloading {file_acc}: {e}. Skipping."
                        )
                        if os.path.exists(dest_path):
                            os.remove(dest_path)
                        continue

                    peak_count = count_gzipped_lines(dest_path)
                    if peak_count < snakemake.config["min_peaks"]:
                        print(
                            f"[REJECTED] {clean_target} ({file_acc}) has only {peak_count} peaks. Deleting."
                        )
                        os.remove(dest_path)
                        continue

                    print(
                        f"[SUCCESS] {clean_target} ({file_acc}) passed with {peak_count} peaks."
                    )
                    success_count += 1
                    downloaded_targets.add(clean_target)
                    found_for_target = True
                    time.sleep(0.1)
                    break

    print(f"[COMPLETE] Downloaded {success_count} valid peak files into '{outdir}'.")

    with open(snakemake.output.sentinel, "w") as f:
        f.write("done")


if __name__ == "__main__":
    main()
