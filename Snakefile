configfile: "config.yaml"

rule all:
    input:
        "data/processed/target_ids.json"

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
