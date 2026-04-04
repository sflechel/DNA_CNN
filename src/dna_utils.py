import numpy as np


def one_hot_encode(sequence: str) -> np.ndarray:
    print(sequence.upper().encode("ascii"))
    byte_array = np.frombuffer(
        sequence.upper().encode("ascii"), dtype=np.uint8
    )  # .encode urns seq into array of bytes
    lookup = np.zeros((256, 4), dtype=np.float32)
    lookup[ord("A")] = [1, 0, 0, 0]
    lookup[ord("C")] = [0, 1, 0, 0]
    lookup[ord("G")] = [0, 0, 1, 0]
    lookup[ord("T")] = [0, 0, 0, 1]
    return lookup[
        byte_array
    ]  # numpy advanced indexing: each value is byte_array serves as an index of lookup, the result is collated into a matrix


if __name__ == "__main__":
    seq = "ACGTNNGCta"
    encoded = one_hot_encode(seq)
    print(f"Encoded shape:{encoded.shape}")
    print(encoded)
