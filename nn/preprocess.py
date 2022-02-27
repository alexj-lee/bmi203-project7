# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
import collections

NUC_MAPPER = {
    "A": [1, 0, 0, 0],
    "T": [0, 1, 0, 0],
    "C": [0, 0, 1, 0],
    "G": [0, 0, 0, 1],
    " ": [0, 0, 0, 0],
}

# Defining preprocessing functions
def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one hot encoding of a list of nucleic acid sequences
    for use as input into a fully connected neural net.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence
            length due to the one hot encoding scheme for nucleic acids.

            For example, if we encode
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    """

    encodings = []
    max_length = len(max(seq_arr, key=len))

    for seq_idx, seq in enumerate(seq_arr):
        seq_upper = seq.upper()
        seq_padded = seq_upper.ljust(max_length)
        encoding = list(
            map(
                NUC_MAPPER.get, iter(seq_padded)
            )  # ljust will pad all strings to correct length
        )  # for each element in seq, replace with the value from NUC_MAPPER
        # will generate a list of lists with only one index hot for each seq
        # if the user passes a string that has characters other than ACTG, NUC_MAPPER.get will return None in the encoding

        assert None not in encoding, f"Got an invalid character in seq # {seq_idx+1}"

        encodings.append(np.array(encoding).flatten())

    return encodings


def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample your sequences to account for class imbalance.
    Consider this as a sampling scheme with replacement.

    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """

    if len(seqs) != len(labels):
        raise ValueError("Length of provided labels and sequences is not the same")

    uniq_labels = collections.defaultdict(list)

    for idx, (seq, label) in enumerate(zip(seqs, labels)):
        if isinstance(label, int) is False:
            raise TypeError("Labels must be of type int")

        uniq_labels[label].append(idx)

        if len(uniq_labels.keys()) > 2:
            raise ValueError("Must be only two classes in dataset")

    undersampled_class = min(uniq_labels, key=uniq_labels.get)
    oversampled_class = [
        key for key in uniq_labels.keys() if key != undersampled_class
    ][0]

    ratio = len(uniq_labels[oversampled_class]) // len(uniq_labels[undersampled_class])

    undersampled_indices = uniq_labels[undersampled_class] * ratio
    new_indices = undersampled_indices + [uniq_labels[oversampled_class]]

    all_data = []
    for ind in undersampled_indices:
        all_data.append((seqs[ind], undersampled_class))

    for ind in uniq_labels[oversampled_class]:
        all_data.append((seqs[ind], oversampled_class))

    # all_data = list(zip(all_data))
    np.random.shuffle(all_data)

    sampled_seqs, sampled_labels = list(zip(*all_data))
    return sampled_seqs, sampled_labels
