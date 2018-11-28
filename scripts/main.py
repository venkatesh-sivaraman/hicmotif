import os, sys
import numpy as np
import rnn
import models
import pickle
from Bio import SeqIO

def load_pickles(filename):
    """
    Iterates over pickles loaded from the given file.
    """
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

def load_data(hic_dir, seq_path, histone_path, seq_format='fasta'):
    """
    Returns a list of (Sequence, InteractionMatrix) tuples given the Hi-C data,
    the DNA sequences, and histone modifications in the paths provided.

    hic_dir: a path to a directory containing files named "loops_<NUM>.pickle",
            where each file contains a series of pickled tuples as written by
            load_submatrices.py
    seq_path: a path to a pickle containing genome subsequences as written by
            load_genome_ranges.py
    histone_path: a path to a pickle containing histone modifications as written
            by epigenome_seq_extraction.py
    """

    sequences = {}
    histones = {}
    # Read histone data
    print("Reading histone data...")
    for id, histone_data in load_pickles(histone_path):
        histones[id.replace('chr', '')] = histone_data

    # Read sequence data
    print("Reading sequence data...")
    for record in SeqIO.parse(seq_path, seq_format):
        id = record.id.replace('chr', '')
        if id not in histones:
            print("Histone modifications not found for sequence {}".format(id))
            continue

        seq = models.Sequence(id, str(record.seq), histones[id])
        sequences[id] = seq

    # Read Hi-C data
    print("Reading Hi-C data...")
    results = {}
    file_idx = 0
    num_omitted = 0
    while True:
        path = os.path.join(hic_dir, "loops_{}.pickle".format(file_idx))
        if not os.path.exists(path):
            break
        for id, data in load_pickles(path):
            if id not in histones:
                print("Sequence object not found for sequence {}".format(id))
                continue
            try:
                results[id] = (sequences[id], models.InteractionMatrix(id, data))
            except ValueError:
                num_omitted += 1
        file_idx += 1

    print("Loaded {} items - {} had missing data.".format(len(results), num_omitted))
    return [val for key, val in sorted(results.items())]

def generate_X(data, seq_length=100, spacing=100):
    """
    Generate an X matrix given a list of tuples (Sequence, InteractionMatrix).
    This matrix uses only the sequences, and the order is preserved with the
    input list. A list of ranges is also returned, where each range is a tuple
    (start, end) that indicates the rows in X corresponding to the given item
    in the input list.
    """
    Xs = []
    ranges = []
    for seq, _ in data:
        seq_array = seq.to_array()
        num_samples = seq_array.shape[1] // spacing
        ranges.append((len(Xs), len(Xs) + num_samples))
        for sample_idx in range(num_samples):
            start_idx = spacing * sample_idx
            Xs.append(seq_array[:,start_idx:start_idx + seq_length])

    return np.concatenate(Xs), ranges

def train_iteration(X, ranges, data, pairwise_interactions=None, rnn_params={}):
    """
    Perform one iteration of training, using the initial pairwise interaction
    matrix to generate a state labeling for each interaction matrix, then training
    an RNN using the X matrix and the inferred state labels.

    X: a numpy array with dimension m x k x n, where m is the number of training
        examples, k is the number of features per base, and n is the sequence
        length.
    ranges: the ranges of rows in X corresponding to each item in data.
    data: a list of tuples (Sequence, InteractionMatrix) as generated from
        load_data.
    pairwise_interactions: the initial pairwise interaction matrix. The shape
        should be l x l, where l indicates the number of states to infer.
    rnn_params: keyword arguments to be passed to the RNNModel constructor.

    Return values TODO
    """
    # TODO
    pass

if __name__ == '__main__':
    data = load_data("../data/GM12878_10k", "../data/loop_sequences_GM12878.fasta", "../data/epigenomic_tracks/GM12878.pickle")
    X, ranges = generate_X(data[:500])
    print(X.shape, ranges)
