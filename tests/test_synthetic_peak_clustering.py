import json
import h5py
import numpy as np
import pytest
from pyfaidx import Fasta

# Replace this import with your actual script entrypoint/function
from scripts.extract_positives import extract_positives


@pytest.fixture
def synthetic_genome_and_bed(tmp_path):
    """Generates a synthetic FASTA (chr1: 10,000 bp) and BED file."""
    fasta_path = tmp_path / "synthetic_genome.fa"
    bed_path = tmp_path / "master.bed"
    targets_path = tmp_path / "targets.json"
    output_h5 = tmp_path / "output.h5"
    output_stats = tmp_path / "output.json"

    # 1. Create Synthetic 10k bp FASTA (random ACGT repeat)
    seq = "ACGT" * 2500
    with open(fasta_path, "w") as f:
        f.write(f">chr1\n{seq}\n")

    # 2. Target mapping JSON
    target_mapping = {
        "target_list": ["CTCF", "RAD21", "MYC"],
        "targets_to_ids": {"CTCF": 0, "RAD21": 1, "MYC": 2},
    }
    with open(targets_path, "w") as f:
        json.dump(target_mapping, f)

    # 3. Create BED file with controlled scenarios:
    # Cluster 1 (co-localized CTCF + RAD21 within 200bp):
    # - chr1: 2000-2100 (CTCF)
    # - chr1: 2200-2300 (RAD21)
    # Cluster 2 (isolated MYC):
    # - chr1: 6000-6100 (MYC)
    bed_content = (
        "chr1\t2000\t2100\tCTCF\nchr1\t2200\t2300\tRAD21\nchr1\t6000\t6100\tMYC\n"
    )
    with open(bed_path, "w") as f:
        f.write(bed_content)

    return {
        "fasta": str(fasta_path),
        "bed": str(bed_path),
        "targets": str(targets_path),
        "output_h5": str(output_h5),
        "output_stats": str(output_stats),
    }


def test_pipeline_label_assignment(synthetic_genome_and_bed):
    """Executes the extraction pipeline and validates targets array in HDF5."""
    data = synthetic_genome_and_bed

    # Run extraction
    extract_positives(
        chrom="chr1",
        window_size=1100,
        inner_size=1000,
        target_ids=data["targets"],
        master_bed=data["bed"],
        fa=data["fasta"],
        h5=data["output_h5"],
        stats=data["output_stats"],
    )

    # Verify output HDF5
    with h5py.File(data["output_h5"], "r") as hf:
        targets = hf["targets"]  # Shape: (N_samples, 3)
        assert isinstance(targets, h5py.Dataset)
        sequences = hf["inputs"]  # Shape: (N_samples, 2000, 4)
        assert isinstance(sequences, h5py.Dataset)

        # Expected 2 samples (Cluster 1 = CTCF+RAD21, Cluster 2 = MYC)
        assert targets.shape[0] == 2
        assert targets.shape[1] == 3

        # Sample 0 (Cluster 1): Must have BOTH CTCF (0) and RAD21 (1) labeled 1.0
        np.testing.assert_array_equal(targets[0], [1.0, 1.0, 0.0])

        # Sample 1 (Cluster 2): Must have ONLY MYC (2) labeled 1.0
        np.testing.assert_array_equal(targets[1], [0.0, 0.0, 1.0])

        # Sequence check (One-hot shape check)
        print(sequences)
        assert sequences.shape[0] == 2
        assert sequences.shape[1] == 4
        assert sequences.shape[2] == 1100
