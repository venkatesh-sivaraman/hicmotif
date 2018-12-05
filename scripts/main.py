import os, sys
import numpy as np
import rnn
import models
import pickle
from Bio import SeqIO
from update_state_assignments import state_asg, normalized_diff, penalty
from scipy import interpolate
import multiprocessing as mp
from functools import partial

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

def load_data(hic_dir, seq_path, histone_path, seq_format='fasta', test=False):
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
        if test and len(histones) == 100: break

    # Read sequence data
    print("Reading sequence data...")
    for record in SeqIO.parse(seq_path, seq_format):
        id = record.id.replace('chr', '')
        if id not in histones:
            print("Histone modifications not found for sequence {}".format(id))
            continue

        seq = models.Sequence(id, str(record.seq), histones[id])
        sequences[id] = seq
        if test and len(sequences) == len(histones): break

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
        if test and len(results) == len(sequences): break

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
    for seq, mat in data:
        seq_array = seq.to_array()
        min_m, max_m = mat.range()
        dim = ((max_m - mat.resolution + 1) - min_m) // spacing
        num_samples = dim - seq_length // spacing #(min(start, seq_array.shape[1]) - seq_length) // spacing - 1
        #print(stop - start, seq_array.shape[1] - seq_length, num_samples)
        ranges.append((len(Xs), len(Xs) + num_samples))
        for sample_idx in range(num_samples):
            start_idx = spacing * sample_idx
            Xs.append(seq_array[:,start_idx:start_idx + seq_length])

    return np.stack(Xs), ranges

def interpolate_state_labels(state_labels, original_spacing, new_spacing):
    """
    Interpolates the given list of state labels from one spacing into another.
    """
    max_x = state_labels.shape[0] * original_spacing
    f = interpolate.interp1d(np.arange(0, max_x, original_spacing), state_labels, kind='nearest')
    return f(np.arange(0, max_x - original_spacing + 1, new_spacing))

def estimate_pairwise_interactions(data, ranges, Y, n_labels, seq_length, spacing):
    """
    Produces a pairwise interaction matrix R of size n_labels x n_labels. Each
    element is the ML estimate of the interaction frequency between the states
    represented by the row and column.
    """
    R = np.zeros((n_labels, n_labels))
    counts = np.zeros((n_labels, n_labels))
    for (start, stop), (_, mat) in zip(ranges, data):
        print(mat.identifier)
        r = mat.range()
        # Get meshgrid for interaction matrix
        loci_x, loci_y, dim = mat.meshgrid(spacing, start_val=(seq_length // spacing) * spacing, max_dim=stop - start)
        if dim > stop - start:
            print("Too large dimension, skipping: {} vs {} - {}".format(dim, stop, start))
            continue

        # Get meshgrid for states
        states_1, states_2 = np.meshgrid(Y[start:start + dim], Y[start:start + dim])
        vals = mat.value_at(loci_x, loci_y)
        if np.sum(np.isnan(vals)) > 0:
            print("Shouldn't happen")

        # Update counts and interaction frequency estimates
        for s1 in range(n_labels):
            for s2 in range(n_labels):
                flags = np.logical_and(states_1 == s1, states_2 == s2)
                counts[s1, s2] += np.sum(flags)
                R[s1, s2] += np.sum(np.where(flags, vals, 0.0))

    return R / counts


def dp_worker(R, coarse_spacing, spacing, seq_length, batch_item):
    (_, H), rnn_labels = batch_item
    print(H.identifier)
    state_labels = state_asg(H, R, normalized_diff, penalty, coarse_spacing, rnn_labels)
    # Save the state labels that begin after the first seq_length region
    trimmed = interpolate_state_labels(state_labels, coarse_spacing, spacing)
    return H.identifier, trimmed[seq_length // spacing:]

def train_iteration(data, R, seq_length=100, spacing=10, rnn_params={}, batch_size=40, old_rnn=None):
    """
    Perform one iteration of training, using the initial pairwise interaction
    matrix to generate a state labeling for each interaction matrix, then training
    an RNN using the X matrix and the inferred state labels.

    data: a list of tuples (Sequence, InteractionMatrix) as generated from
        load_data.
    R: the initial pairwise interaction matrix. The shape should be l x l, where
        l indicates the number of states to infer.
    rnn_params: keyword arguments to be passed to the RNNModel constructor.
    batch_size: the number of elements of data to select randomly to use in this
        iteration.
    old_rnn: an RNNModel object, if available, whose predictions will be used
        to bias the new DP algorithm's predicted state labels.

    Returns:
        * an RNNModel object trained on a batch of the given data
        * a new R matrix based on the RNNModel's predictions
    """
    n_labels = R.shape[0]

    # Generate a random batch of data
    indexes = np.random.choice(len(data), size=(batch_size,), replace=False)
    batch = [data[i] for i in indexes]
    X, ranges = generate_X(batch, seq_length=seq_length, spacing=spacing)

    print(X.shape)
    # Run DP algorithm
    coarse_spacing = spacing * 10
    Y_single = np.zeros((X.shape[0],), dtype=int) # Each element corresponds to one row of X
    pool = mp.Pool(processes=4)
    worker = partial(dp_worker, R, coarse_spacing, spacing, seq_length)
    # Vectors corresponding to states for each interaction matrix
    Y_pred = old_rnn.predict(X) if old_rnn is not None else None
    rnn_iter = (Y_pred[start:stop] for start, stop in ranges) if old_rnn is not None else (None for _ in ranges)
    print("Prediction composition:", np.unique(Y_pred, return_counts=True))

    for i, (id, trimmed) in enumerate(pool.imap(worker, zip(batch, rnn_iter), chunksize=2)):
        start, stop = ranges[i]
        #TODO: Don't allow these cases to pass
        if trimmed.shape[0] < stop - start:
            print("Too big:", trimmed.shape[0], start, stop, stop - start)
            continue
        # Store predicted state labels in Y vector
        Y_single[start:stop] = trimmed[:stop - start]

    pool.close()
    pool.join()

    # Convert Y_single to one-hot representation
    Y = np.zeros((X.shape[0], n_labels), dtype=int)
    for i in range(Y_single.shape[0]):
        Y[i,Y_single[i]] = 1

    print(X.shape, Y.shape)

    model = rnn.RNNModel(n_labels=n_labels, n_features=X.shape[1], sequence_length=seq_length, **rnn_params)
    model.create()
    model.train(X, Y, epochs=10)
    Y_pred = model.predict(X)
    print("Prediction composition on previously-trained:", np.unique(Y_pred, return_counts=True))

    return model, estimate_pairwise_interactions(batch, ranges, Y_pred, n_labels, seq_length, spacing)

def initial_pairwise_interactions(data, n_labels, batch_size=10, bound=0.25):
    """
    Generates an random initial R matrix of dimension n_labels x n_labels given
    the data.

    data: a list of tuples (Sequence, InteractionMatrix) as generated from
        load_data.
    batch_size: the number of elements of data to select randomly to use in this
        iteration.
    """
    def shear_matrix(ssm):
        # Each column shifts one more down
        new_ssm = np.zeros(ssm.shape)
        for n in range(ssm.shape[0]):
            new_ssm[:,n] = np.roll(ssm[:,n], -n)
        return new_ssm

    batch_H = [shear_matrix(data[i][1].values_grid()) for i in range(batch_size)]
    diagonal_avg = 0
    off_diagonal_avg = 0
    for i in range(len(batch_H)):
        H = batch_H[i]
        bound_len = int(bound*H.shape[0])
        diagonal_avg += np.average(np.concatenate((H[:bound_len//2,:], H[H.shape[0] - bound_len//2:,:])))
        off_diagonal_avg += np.average(H[bound_len//2:H.shape[0] - bound_len//2,:])

    diagonal_avg /= len(batch_H)
    off_diagonal_avg /= len(batch_H)

    R = np.full((n_labels,n_labels), off_diagonal_avg, dtype=float)
    np.fill_diagonal(R, diagonal_avg)
    return R

if __name__ == '__main__':
    test = len(sys.argv) > 1 and sys.argv[1] == "test"
    data = load_data("../data/GM12878_10k", "../data/loop_sequences_GM12878.fasta", "../data/epigenomic_tracks/GM12878.pickle", test=test)
    seq_length = 100
    spacing = 50 if test else 10
    Rs = [initial_pairwise_interactions(data, 2)]
    old_rnn = None
    print("Initial:", Rs[-1])
    for i in range(10):
        print("Iteration", i)
        new_rnn, R = train_iteration(data, Rs[-1], batch_size=(20 if test else 40), seq_length=seq_length, spacing=spacing, old_rnn=old_rnn)
        Rs.append(R)
        if not os.path.exists("../data/rnns"): os.mkdir("../data/rnns")
        new_rnn.model.save("../data/rnns/iteration_{}.hd5".format(i))
        old_rnn = new_rnn
        print(Rs[-1])
    print(Rs)
