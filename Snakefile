configfile: "config.yaml"

CHROMS=[f"chr{i}" for i in range(1,23)] + ["chrX", "chrY"]

rule all:
    input:
        train_h5="data/processed/dataset_train.h5",
        val_h5="data/processed/dataset_validation.h5",
        test_h5="data/processed/dataset_test.h5"

rule merge_tensors:
    input:
        pos_h5s=expand("data/processed/tensors/{chrom}_pos.h5", chrom=CHROMS),
        neg_h5s=expand("data/processed/tensors/{chrom}_neg.h5", chrom=CHROMS),
        target_ids="data/processed/target_ids.json"
    output:
        train_h5="data/processed/dataset_train.h5",
        val_h5="data/processed/dataset_validation.h5",
        test_h5="data/processed/dataset_test.h5"
    script:
        "scripts/merge_tensors.py"


rule extract_negatives
    input:
        fa="data/fasta/hg38.fa",
        master_bed="data/processed/master.bed",
        pos_stats="data/processed/tensors/{chrom}_stats.json",
        target_ids="data/processed/target_ids.json"
    output:
        h5="data/processed/tensors/{chrom}_neg.h5"
    script:
        "scripts/extract_negatives.py"

rule extract_positives:
    input:
        fa="data/fasta/hg38.fa",
        master_bed="data/processed/master.bed",
        target_ids="data/processed/target_ids.json"
    output:
        h5="data/processed/tensors/{chrom}_pos.h5",
        stats="data/processed/tensors/{chrom}_stats.json"
    script:
        "scripts/extract_positives.py"

rule map_targets:
    input:
        master_bed="data/processed/master.bed"
    output:
        target_ids="data/processed/target_ids.json"
    script:
        "scripts/map_targets_to_id.py"

rule merge_beds:
    input:
        sentinel="data/beds/.download_complete"
    output:
        master_bed="data/processed/master.bed"
    script:
        "scripts/merge_beds.py"

rule download_fasta:
    output:
        fa="data/fasta/hg38.fa"
    script:
        "scripts/download_fasta.py"

rule download_beds:
    output:
        sentinel="data/beds/.download_complete"
    script:
        "scripts/download_beds.py"
