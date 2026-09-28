import json
import h5py
import numpy as np
import pytest

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
    seq_list = list("ACGT" * 2500)
    # Embed distinctive nucleotide motifs at specific genomic coordinates:
    # CTCF peak 1A & 1B: Poly-A motif at chr1: 2010-2040
    seq_list[2010:2040] = list("A" * 30)
    # RAD21 peak: Poly-T motif at chr1: 2210-2240
    seq_list[2210:2240] = list("T" * 30)
    # MYC peak: Poly-C motif at chr1: 6010-6040
    seq_list[6010:6040] = list("C" * 30)
    seq = "".join(seq_list)
    with open(fasta_path, "w") as f:
        f.write(f">chr1\n{seq}\n")

    # 2. Target mapping JSON
    target_mapping = {
        "target_list": ["CTCF", "RAD21", "MYC"],
        "targets_to_ids": {"CTCF": 0, "RAD21": 1, "MYC": 2},
    }
    with open(targets_path, "w") as f:
        json.dump(target_mapping, f)

    bed_content = (
        "chr1\t2000\t2100\tCTCF\nchr1\t2200\t2300\tRAD21\nchr1\t6000\t6100\tMYC\n"
    )
    # 3. Create BED file testing two scenarios:
    # Cluster 1 (co-localized & duplicate peaks within 1000bp inner_size):
    # - chr1: 2000-2050 (CTCF Peak 1)
    # - chr1: 2060-2100 (CTCF Peak 2 -> Duplicate/Nearby same TF)
    # - chr1: 2200-2250 (RAD21 Peak)
    # Cluster 2 (isolated MYC peak):
    # - chr1: 6000-6050 (MYC Peak)
    bed_content = (
        "chr1\t2000\t2050\tCTCF\n"
        "chr1\t2060\t2100\tCTCF\n"  # Nearby same-TF peak
        "chr1\t2200\t2250\tRAD21\n"
        "chr1\t6000\t6050\tMYC\n"
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


def test_pipeline_label_assignment_and_sequence_motifs(synthetic_genome_and_bed):
    """Verifies:

    1. Correct multi-hot label assignment.
    2. Proper merging of same-TF duplicate peaks into a single cluster.
    3. Accurate one-hot encoded sequence motifs embedded at relative coordinates.
    """
    data = synthetic_genome_and_bed
    window_size = 1100

    # Execute extraction
    extract_positives(
        chrom="chr1",
        window_size=window_size,
        inner_size=1000,
        target_ids=data["targets"],
        master_bed=data["bed"],
        fa=data["fasta"],
        h5=data["output_h5"],
        stats=data["output_stats"],
    )

    # Validate HDF5 outputs
    with h5py.File(data["output_h5"], "r") as hf:
        targets = hf["targets"]  # Shape: (N_samples, 3)
        inputs = hf["inputs"]  # Shape: (N_samples, 4, window_size)

        assert isinstance(targets, h5py.Dataset)
        assert isinstance(inputs, h5py.Dataset)

        # Expected 2 clusters (Cluster 1 merged 3 peaks into 1 sample)
        assert targets.shape == (2, 3)
        assert inputs.shape == (2, 4, window_size)

        # --- Check Labels & Peak Merging ---
        # Sample 0 (Cluster 1): CTCF (0) + RAD21 (1) -> [1.0, 1.0, 0.0]
        # (Duplicate CTCF peaks merged cleanly without double counting)
        np.testing.assert_array_equal(targets[0], [1.0, 1.0, 0.0])

        # Sample 1 (Cluster 2): MYC (2) -> [0.0, 0.0, 1.0]
        np.testing.assert_array_equal(targets[1], [0.0, 0.0, 1.0])

        # --- Check Sequence Motifs in One-Hot Output ---
        # Assuming channel mapping: 0 -> A, 1 -> C, 2 -> G, 3 -> T

        # Cluster 1 Bounds: start=2000, end=2250 -> center=2125
        # Extracted window: 2125 - 550 = 1575 to 2125 + 550 = 2675
        cluster1_seq_start = 1575

        # Poly-A motif (CTCF) at genomic 2010..2040 -> Relative offset: 435..465
        ctcf_rel_start = 2010 - cluster1_seq_start
        ctcf_rel_end = 2040 - cluster1_seq_start
        # All channels at row 0 (A) in this region must be 1.0
        assert np.all(inputs[0, 0, ctcf_rel_start:ctcf_rel_end] == 1.0)

        # Poly-T motif (RAD21) at genomic 2210..2240 -> Relative offset: 635..665
        rad21_rel_start = 2210 - cluster1_seq_start
        rad21_rel_end = 2240 - cluster1_seq_start
        # All channels at row 3 (T) in this region must be 1.0
        assert np.all(inputs[0, 3, rad21_rel_start:rad21_rel_end] == 1.0)

        # Cluster 2 Bounds: start=6000, end=6050 -> center=6025
        # Extracted window: 6025 - 550 = 5475 to 6025 + 550 = 6575
        cluster2_seq_start = 5475

        # Poly-C motif (MYC) at genomic 6010..6040 -> Relative offset: 535..565
        myc_rel_start = 6010 - cluster2_seq_start
        myc_rel_end = 6040 - cluster2_seq_start
        # All channels at row 1 (C) in this region must be 1.0
        assert np.all(inputs[1, 1, myc_rel_start:myc_rel_end] == 1.0)


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
