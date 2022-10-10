import functools
import itertools
import tensorflow as tf
import numpy as np
class ConcatenateSequence(tf.keras.utils.Sequence):
    def __init__(self, sequences, batch_size):
        self.batch_size = batch_size
        assert all(seq.batch_size == batch_size for seq in sequences), f"Batch size must be equal to {batch_size} for all sequences"
        self.sequences = sequences
        part_lengths = [len(sequence) for sequence in self.sequences]
        parts = list(itertools.accumulate(part_lengths))
        self.bins = np.array(parts)
        self.length = sum(part_lengths)
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        part = np.digitize(idx, self.bins)
        if part != 0:
            part_idx = idx - self.bins[part - 1]
        else:
            part_idx = idx
        return self.sequences[part][part_idx]
