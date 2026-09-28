import json
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
bed_path = snakemake.input.master_bed
targets: set[str] = set()

with open(bed_path, "r") as file:
    for line in file:
        parts: list[str] = line.strip().split()
        if len(parts) < 4:
            continue
        targets.add(parts[3])

target_list: list[str] = sorted(list(targets))
targets_to_ids: dict[str, int] = {name: id for id, name in enumerate(target_list)}
logging.info(f"Found {len(target_list)} unique targets. Mapping them to unique ids")

os.makedirs(os.path.dirname(snakemake.output.target_ids), exist_ok=True)
with open(snakemake.output.target_ids, "w") as file:
    json.dump({"target_list": target_list, "targets_to_ids": targets_to_ids}, file)
logging.info(f"Wrote target list and target ids to {snakemake.output.target_ids}")
