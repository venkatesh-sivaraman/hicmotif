import numpy as np
import pickle
from scipy import interpolate

SEQUENCE_ORDER = {'A': 0, 'G': 1, 'T': 2, 'C': 3}
NUM_HISTONE_MODS = 5

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
        mod_arrays = np.zeros((NUM_HISTONE_MODS, len(self.seq)))
        for histone, start, end in self.histone_mods:
            mod_arrays[histone,start:end] = 1
        return mod_arrays

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
            if c not in SEQUENCE_ORDER: continue
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
        * throws a ValueError if the difference between all available loci is
          not equal to the inferred resolution
        """
        self.identifier = identifier
        xs = sorted(set(x for x, y, f in data))
        self.data = {(x, y): f for x, y, f in data}
        # Determine the resolution of the data
        self.resolution = xs[1] - xs[0]
        if not all(xs[i] - xs[i - 1] == self.resolution for i in range(1, len(xs))):
            raise ValueError("missing data: distances between available loci do not match inferred resolution")
        self.build_interpolator()

    def range(self):
        """
        Returns the range of for which data is available in this matrix, as a
        tuple (min, max).
        """
        xs = sorted(set(x for x, y in self.data.keys()))
        return (min(xs), max(xs) + self.resolution)

    def build_interpolator(self):
        """
        Builds an interpolator for this interaction matrix. Values that are
        not present are omitted from the input to the interpolator, so they will
        not interfere with approximation of those missing values.
        """
        r = self.range()
        loci_x, loci_y = np.meshgrid(np.arange(r[0], r[1], self.resolution), np.arange(r[0], r[1], self.resolution))
        loci_coords = []
        loci_values = []
        for x, y in zip(loci_x.ravel(), loci_y.ravel()):
            # Find the value in the data source
            near_x = int(x / self.resolution) * self.resolution
            near_y = int(y / self.resolution) * self.resolution
            if (near_x, near_y) not in self.data:
                near_x, near_y = near_y, near_x
            if (near_x, near_y) not in self.data:
                continue

            # Add value to interpolator if present
            val = self.data[(near_x, near_y)]
            if val != 0.0:
                loci_coords.append([x, y])
                loci_values.append(val)

        # Build linear interpolator
        self.interpolator = interpolate.LinearNDInterpolator(loci_coords, np.log(1 + np.array(loci_values)))

    def meshgrid(self, resolution=None, start_val=None, max_dim=None):
        """
        Helper function that returns two grids of coordinates for which data is
        available in this interaction matrix, at the given resolution. If
        resolution is None, the matrix's native resolution is used.

        Returns: a tuple containing the x coordinates in a grid, the y coordinates
            in the grid, and dim, the number of values along each side of the
            grid square
        """
        if resolution is None:
            resolution = self.resolution
        r = self.range()
        lower_bound = r[0]
        if start_val is not None:
            lower_bound += start_val
        upper_bound = r[1] - self.resolution + 1
        if max_dim is not None:
            upper_bound = min(upper_bound, lower_bound + max_dim * resolution)
        intervals = np.arange(lower_bound, upper_bound, resolution)
        loci_x, loci_y = np.meshgrid(intervals, intervals)
        return loci_x, loci_y, intervals.shape[0]

    def coordinates_grid(self, resolution=None):
        """
        Helper function that returns a grid of coordinates for which data is
        available in this interaction matrix, at the given resolution. If
        resolution is None, the matrix's native resolution is used.

        Returns: a tuple containing coordinates as a matrix of dimension n x 2
            (n is the total number of points, dim ** 2), and dim, the number of
            values along each side of the grid square
        """
        loci_x, loci_y, dim = self.meshgrid(resolution)
        return np.hstack([loci_x.reshape(-1, 1), loci_y.reshape(-1, 1)]), dim

    def values_grid(self, resolution=None):
        """
        Helper function that returns a grid of (interpolated) values in the Hi-C
        map at the given resolution. If resolution is None, the matrix's native
        resolution is used.

        Returns: Hi-C interaction values as a matrix of dimension n x n, where n
            is the number of steps along this interaction matrix for which data
            is available.
        """
        grid, dim = self.coordinates_grid(resolution)
        return self.value_at(grid[:,0], grid[:,1]).reshape(dim, dim)

    def raw_value_at(self, x, y):
        """
        Returns the raw (un-interpolated) value at the given locus.
        """
        near_x = int(x / self.resolution) * self.resolution
        near_y = int(y / self.resolution) * self.resolution
        if (near_x, near_y) not in self.data:
            near_x, near_y = near_y, near_x
        if (near_x, near_y) not in self.data:
            print("data for {},{} not found in interaction matrix ({},{})".format(x, y, near_x, near_y))
            return 0
        return self.data[(near_x, near_y)]

    def value_at(self, x, y):
        """
        Returns the interaction matrix value at the given locus. x and y may
        be scalars or same-length vectors.
        """
        return self.interpolator(x, y)
