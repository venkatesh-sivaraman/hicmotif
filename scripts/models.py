import numpy as np
import pickle

SEQUENCE_ORDER = {'A': 0, 'G': 1, 'T': 2, 'C': 3}

"""
The Sequence class encapsulates 1-D information on a region of DNA: the genome
sequence as well as a series of histone modifications as binary vectors.
"""
class Sequence:
    def __init__(self, identifier, dna_seq, histone_mods):
        """
        Initialize a new Sequence with the given DNA sequence and histone
        modifications.
        identifier: a string identifier of the form 'chr:start:end'
        dna_seq: a str containing the DNA sequence of length n
        histone_mods: a list of tuples (i, start, end), where i is the index of
            the histone modification type (in a prespecified order), and start
            and end are loci relative to the start of the DNA sequence.
        """
        self.identifier = identifier
        self.seq = dna_seq
        self.histone_mods = histone_mods

    def process_histone_modifications(self):
        """
        Returns a numpy array of size k x n representing the histone modifications
        to the sequence.
        """
        mod_arrays = []
        for histone, start, end in self.histone_mods:
            while histone >= len(mod_arrays):
                mod_arrays.append(np.zeros((len(self.seq),)))
            mod_arrays[histone][start:end] = 1
        return np.array(mod_arrays)

    def range(self):
        """Returns the range of loci for which data is available in this Sequence."""
        chr, start, end = self.identifier.split(':')
        return (int(start), int(end))

    def chromosome(self):
        """Returns the chromosome that this Sequence is on."""
        return self.identifier.split(':')[0]

    def to_array(self):
        """
        Produces a numpy array of shape (k + 4) x n representing the sequence, where k
        is the number of histone features and n is the length of the sequence.
        """
        dna_mat = np.zeros((4, len(self.seq)))
        for i, c in enumerate(self.seq):
            dna_mat[SEQUENCE_ORDER[c], i] = 1

        return np.vstack([dna_mat, self.process_histone_modifications()])

"""
Describes an interaction matrix such as could be produced by Hi-C.
"""
class InteractionMatrix:
    def __init__(self, identifier, data):
        """
        Initializes a new InteractionMatrix with the given data.
        identifier: a string identifier (may be of the form 'chr:start:end')
        data: a list of tuples (x, y, val) where x and y are loci, and val is
            the value of the interaction matrix
        """
        self.identifier = identifier
        xs = sorted(set(x for x, y, f in data))
        self.data = {(x, y): f for x, y, f in data}
        # Determine the resolution of the data
        self.resolution = xs[1] - xs[0]
        assert all(xs[i] - xs[i - 1] == self.resolution for i in range(1, len(xs)))

    def range(self):
        """
        Returns the range of for which data is available in this matrix, as a
        tuple (min, max).
        """
        xs = sorted(set(x for x, y in self.data.keys()))
        return (min(xs), max(xs) + self.resolution)

    def value_at(self, x, y):
        """
        Returns the interaction matrix value at the given locus.
        """
        near_x = int(x / self.resolution) * self.resolution
        near_y = int(y / self.resolution) * self.resolution
        if (near_x, near_y) not in self.data:
            near_x, near_y = near_y, near_x

        if (near_x, near_y) not in self.data:
            print("data for {},{} not found in interaction matrix ({},{})".format(x, y, near_x, near_y))
            return 0
        return self.data[(near_x, near_y)]
