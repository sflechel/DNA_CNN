configfile: "config.yaml"

rule all:
    input:
        "data/fasta/hg38.fa",
        "data/beds/.download_complete"

rule download_fasta:
    output:
        fa="data/fasta/hg38.fa"
    script:
        "scripts/download_fasta.py"

rule download_beds:
    input:
        # Optional: if you want the config file changes to trigger re-checks
        config="config.yaml"
    output:
        sentinel="data/beds/.download_complete"
    script:
        "scripts/download_beds.py"
