import os, sys
import numpy as np
#import rnn
import models
import pickle
from Bio import SeqIO
from update_state_assignments import state_asg, normalized_diff, penalty
from scipy import interpolate
import multiprocessing as mp
from functools import partial

FRACTION_OLD_R = 0.25
CONVERGENCE_THRESHOLD = 0.05
CONVERGENCE_NUM_ITERS = 5
NUM_WORKERS = 4

def adaptive_learning_rate(new, old):
    """
    Computes a sigmoid adaptive learning rate given the new score and the
    old score.
    """
    a = 0.95 # Value of curve at 0
    coef = 1.0 / old * np.log(a / (1.0 - a))
    return 1.0 / (1.0 + np.exp(coef * (new - old)))
    #return 1.0 - FRACTION_OLD_R #0.9 if new < old else 0.6


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

def estimate_worker(Y, n_labels, seq_length, spacing, info):
    (start, stop), (_, mat) = info
    R = np.zeros((n_labels, n_labels))
    counts = np.zeros((n_labels, n_labels))

    print(mat.identifier)
    r = mat.range()
    # Get meshgrid for interaction matrix
    loci_x, loci_y, dim = mat.meshgrid(spacing, start_val=(seq_length // spacing) * spacing, max_dim=stop - start)
    if dim > stop - start:
        print("Too large dimension, skipping: {} vs {} - {}".format(dim, stop, start))
        return None, None
    vals = mat.value_at(loci_x, loci_y)

    # Get meshgrid for states
    states_1, states_2 = np.meshgrid(Y[start:start + dim], Y[start:start + dim])
    if np.sum(np.isnan(vals)) > 0:
        print("Shouldn't happen")

    # Update counts and interaction frequency estimates
    for s1 in range(n_labels):
        for s2 in range(n_labels):
            flags = np.logical_and(states_1 == s1, states_2 == s2)
            counts[s1, s2] += np.sum(flags)
            R[s1, s2] += np.sum(np.where(flags, vals, 0.0))
    return R, counts


def estimate_pairwise_interactions(data, ranges, Y, n_labels, seq_length, spacing):
    """
    Produces a pairwise interaction matrix R of size n_labels x n_labels. Each
    element is the ML estimate of the interaction frequency between the states
    represented by the row and column.
    """
    R = np.zeros((n_labels, n_labels))
    counts = np.zeros((n_labels, n_labels))

    # worker = partial(estimate_worker, Y, n_labels, seq_length, spacing)
    # pool = mp.Pool(processes=NUM_WORKERS)
    # for sub_R, sub_counts in pool.imap(worker, zip(ranges, data)):
    #     if sub_R is None:
    #         continue
    #     R += sub_R
    #     counts += sub_counts

    for (start, stop), (_, mat) in zip(ranges, data):
        print(mat.identifier)
        r = mat.range()
        # Get meshgrid for interaction matrix
        loci_x, loci_y, dim = mat.meshgrid(spacing, start_val=(seq_length // spacing) * spacing, max_dim=stop - start)
        if dim > stop - start:
            print("Too large dimension, skipping: {} vs {} - {}".format(dim, stop, start))
            continue
        vals = mat.value_at(loci_x, loci_y)

        # Get meshgrid for states
        states_1, states_2 = np.meshgrid(Y[start:start + dim], Y[start:start + dim])
        if np.sum(np.isnan(vals)) > 0:
            print("Shouldn't happen")

        # Update counts and interaction frequency estimates
        for s1 in range(n_labels):
            for s2 in range(n_labels):
                flags = np.logical_and(states_1 == s1, states_2 == s2)
                counts[s1, s2] += np.sum(flags)
                R[s1, s2] += np.sum(np.where(flags, vals, 0.0))

    return R / counts


def dp_worker(R, coarse_spacing, spacing, seq_length, batch_item, trim=True):
    (_, H), rnn_labels = batch_item
    score, state_labels = state_asg(H, R, normalized_diff, penalty, coarse_spacing, rnn_labels)
    print(H.identifier, score)
    # Save the state labels that begin after the first seq_length region
    trimmed = interpolate_state_labels(state_labels, coarse_spacing, spacing)
    return H.identifier, score, (trimmed[seq_length // spacing:] if trim else trimmed)

def train_iteration(data, R, seq_length=100, spacing=10, rnn_params={}, batch_size=40, last_score=None):
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
    last_score: the last average score, or None if this is the first iteration.

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
    pool = mp.Pool(processes=NUM_WORKERS)
    worker = partial(dp_worker, R, coarse_spacing, spacing, seq_length)
    scores = []

    for i, (id, score, trimmed) in enumerate(pool.imap(worker, zip(batch, (None for _ in ranges)), chunksize=2)):
        start, stop = ranges[i]
        #TODO: Don't allow these cases to pass
        if trimmed.shape[0] < stop - start:
            print("Too big:", trimmed.shape[0], start, stop, stop - start)
            continue
        # Store predicted state labels in Y vector
        Y_single[start:stop] = trimmed[:stop - start]
        scores.append(score / ((stop - start) * spacing))

    pool.close()
    pool.join()

    # Convert Y_single to one-hot representation
    Y = np.zeros((X.shape[0], n_labels), dtype=int)
    for i in range(Y_single.shape[0]):
        Y[i,Y_single[i]] = 1

    print(X.shape, Y.shape)
    print("Prediction composition on previously-trained:", np.unique(Y_single, return_counts=True))
    new_score = np.nanmean(np.array(scores))
    if last_score is None:
        fraction_new_R = 0.0
    else:
        fraction_new_R = adaptive_learning_rate(new_score, last_score)
        print("Scores:", new_score, last_score, fraction_new_R)

    new_R = estimate_pairwise_interactions(batch, ranges, Y_single, n_labels, seq_length, spacing)
    return new_score, np.where(np.isnan(new_R), R, (1.0 - fraction_new_R) * R + fraction_new_R * new_R)

def initial_pairwise_interactions(data, n_labels, batch_size=10, bound=0.25):
    """
    Generates an random initial R matrix of dimension n_labels x n_labels given
    the data.

    data: a list of tuples (Sequence, InteractionMatrix) as generated from
        load_data.
    batch_size: the number of elements of data to select randomly to use in this
        iteration.
    """
    if int(np.log2(n_labels)) == np.log2(n_labels):
        # Build a hierarchical matrix
        R = np.zeros((1, 1))
        scales = int(np.log2(n_labels))
        for i in range(scales):
            R = np.repeat(np.repeat(R, 2, axis=0), 2, axis=1)
            R += np.diag(np.random.randint(4.0, 6.0, size=2 ** (i + 1)) / scales)
            R += (np.random.random(size=R.shape) * 2.0 + 1.0) / scales
            print(i, R)
        return R
    else:
        R = np.diag(np.random.randint(2.0, 4.0, size=n_labels))
        return R + np.random.random(size=(n_labels, n_labels)) * 2.0 + 1.0

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
    base = "../data/"
    data = load_data(base + "GM12878_5k", base + "loop_sequences_GM12878.fasta", base + "epigenomic_tracks/GM12878.pickle", test=test)
    seq_length = 100
    spacing = 50# if test else 10
    dim = int(sys.argv[-1]) if len(sys.argv) > (2 if test else 1) else 2
    Rs = [initial_pairwise_interactions(data, dim)]
    old_score = None
    print("Initial:", Rs[-1])
    i = 0
    num_unchanging = 0
    while num_unchanging < CONVERGENCE_NUM_ITERS:
        print("Iteration", i)
        new_score, R = train_iteration(data, Rs[-1], batch_size=(20 if test else 60), seq_length=seq_length, spacing=spacing, last_score=old_score)
        Rs.append(R)
        #if not os.path.exists("../data/rnns"): os.mkdir("../data/rnns")
        #new_rnn.model.save("../data/rnns/iteration_{}.hd5".format(i))
        old_score = new_score
        print(Rs[-1])
        delta = np.nanmean(np.abs(Rs[-1] - Rs[-2]))
        print("Delta:", delta)
        if delta < CONVERGENCE_THRESHOLD:
            num_unchanging += 1
        else:
            num_unchanging = 0
        i += 1
    print(Rs)

    pid = os.getpid()
    print("Writing to PID", pid)
    out_path = base + "dp_assignments_mse/assignments_{}.csv".format(pid)
    with open(out_path, 'w') as file:
        pool = mp.Pool(processes=4)
        worker = partial(dp_worker, Rs[-1], spacing * 10, spacing, seq_length, trim=False)
        rnn_iter = (None for _ in data)
        for i, (id, score, state_labels) in enumerate(pool.imap(worker, zip(data, rnn_iter), chunksize=10)):
            file.write(','.join([id, str(score)] + [str(label) for label in state_labels]) + '\n')
