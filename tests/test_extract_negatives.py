import pytest
from scripts.extract_negatives import (
    build_exclusion_zones,
    is_excluded,
    find_and_pop_gc_match,
)


# we assume peaks are sorted by start position, and that end is great than start
def test_build_exclusion_zones():
    peaks: list[tuple[int, int, int]] = [(1000, 1200, 0), (2000, 2300, 1)]
    window_size: int = 100

    expected: list[tuple[int, int]] = [(900, 1300), (1900, 2400)]
    assert build_exclusion_zones(peaks, window_size) == expected


def test_build_exclusion_zones_merging():
    peaks = [(1000, 1200, 0), (1250, 1400, 2), (2000, 2300, 1)]
    window_size: int = 100

    expected = [(900, 1500), (1900, 2400)]
    assert build_exclusion_zones(peaks, window_size) == expected


def test_is_excluded():
    zones: list[tuple[int, int]] = [(900, 1300), (1900, 2400)]
    not_excluded: tuple[int, int] = (1400, 1600)
    excluded1: tuple[int, int] = (1200, 1400)
    excluded2: tuple[int, int] = (1700, 1950)
    excluded3: tuple[int, int] = (800, 1400)
    edges_touching: tuple[int, int] = (1300, 1400)

    assert is_excluded(not_excluded[0], not_excluded[1], zones) is False
    assert is_excluded(excluded1[0], excluded1[1], zones) is True
    assert is_excluded(excluded2[0], excluded2[1], zones) is True
    assert is_excluded(excluded3[0], excluded3[1], zones) is True
    # if edges are touching, there is no actual overlap!
    assert is_excluded(edges_touching[0], edges_touching[1], zones) is False


def test_find_and_pop_gc_match():
    gcs: list[float] = [0.1, 0.5, 0.9]
    bad_candidate: float = 0.3
    good_candidate: float = 0.11
    tolerance: float = 0.03

    bad_match = find_and_pop_gc_match(bad_candidate, gcs, tolerance)
    assert bad_match is None
    assert gcs == [0.1, 0.5, 0.9]

    good_match = find_and_pop_gc_match(good_candidate, gcs, tolerance)
    assert good_match == 0.1
    assert gcs == [0.5, 0.9]


def test_find_and_pop_gc_match_empty_list():
    gcs = []
    candidate = 0.3
    tolerance = 0.03

    match = find_and_pop_gc_match(candidate, gcs, tolerance)
    assert match is None
    assert not gcs
