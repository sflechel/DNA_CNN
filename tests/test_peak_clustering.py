import pytest
from scripts.extract_positives import cluster_peaks


def test_single_peak_clustering():
    # Format: (start, end, tf_id)
    peaks = [(100, 200, 1)]
    clusters = cluster_peaks(peaks, max_span=1000)

    assert len(clusters) == 1
    assert clusters[0] == [(100, 200, 1)]


def test_colocalized_peaks_merge():
    """Two TFs 200 bp apart should merge into 1 cluster."""
    peaks = [(1000, 1100, 1), (1200, 1300, 2)]
    clusters = cluster_peaks(peaks, max_span=1000)

    assert len(clusters) == 1
    assert len(clusters[0]) == 2
    assert clusters[0][0][2] == 1
    assert clusters[0][1][2] == 2


def test_distant_peaks_split():
    """Peaks further apart than max_span must be placed in separate clusters."""
    peaks = [(1000, 1100, 1), (2500, 2600, 2)]
    clusters = cluster_peaks(peaks, max_span=1000)

    assert len(clusters) == 2
    assert clusters[0] == [(1000, 1100, 1)]
    assert clusters[1] == [(2500, 2600, 2)]


def test_chaining_exceeding_max_span():
    """
    Peak 1 (1000-1100) and Peak 2 (1400-1500) span 500bp.
    Peak 3 (2100-2200) is 600bp from Peak 2, but total span 1000->2200 is 1200bp (> max_span).
    Should split Peak 3 into Cluster 2.
    """
    peaks = [(1000, 1100, 1), (1400, 1500, 2), (2100, 2200, 3)]
    clusters = cluster_peaks(peaks, max_span=1000)

    assert len(clusters) == 2
    assert len(clusters[0]) == 2  # TF1 & TF2
    assert len(clusters[1]) == 1  # TF3
