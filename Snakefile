configfile: "config.yaml"

CHROMS=[f"chr{i}" for i in range(1,23)] + ["chrX", "chrY"]

rule all:
    input:
        pos_h5s=expand("data/processed/tensors/{chrom}_pos.h5", chrom=CHROMS)

rule extract_positives:
    input:
        fa="data/fasta/hg38.fa",
        master_bed="data/processed/master.bed",
        target_ids="data/processed/target_ids.json"
    output:
        h5="data/processed/tensors/{chrom}_pos.h5",
        stats="data/processed/tensors/{chrom}_stats.h5"
    params:
        window_size=1100,
        inner_size=1100
    script:
        "scripts/extract_positives.py"

rule map_targets:
    input:
        master_bed="data/processed/master.bed"
    output:
        target_ids="data/processed/target_ids.json"
    script:
        "scripts/map_targets_to_id.py"

rule merge_all:
    input:
        sentinel="data/beds/.download_complete"
    output:
        master_bed="data/processed/master.bed"
    params:
        beds_dir="data/beds"
    script:
        "scripts/merge_beds.py"

rule download_fasta:
    output:
        fa="data/fasta/hg38.fa"
    script:
        "scripts/download_fasta.py"

rule download_beds:
    input:
        config="config.yaml"
    output:
        sentinel="data/beds/.download_complete"
    script:
        "scripts/download_beds.py"
