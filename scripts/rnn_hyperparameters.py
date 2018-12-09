import numpy as np
import os
import random
import pickle
import pandas as pd
from Bio import SeqIO
import matplotlib.pyplot as plt
import sys
import models
import rnn
from main import *

def evaluate_consistency(mat, states, resolution, num_states):
    """
    Returns a measure of consistency of the given state mapping with respect to the interaction matrix.
    mat - an InteractionMatrix object
    states - a state labeling over the sequence of the given matrix
    resolution - the resolution of the given state labeling (i.e. number of base pairs labeled per element of states)
    num_states - the number of states possible
    """
    # Collect interaction frequencies
    interaction_frequencies = [[[] for i in range(num_states)] for j in range(num_states)]
    start, end = mat.range()
    _, num_data_points = mat.coordinates_grid(resolution)
    #print(len(states), "vs", num_data_points)
    values = np.log(1 + mat.values_grid(resolution))

    # Group interaction frequencies by which pair of states were assigned to x and y
    for i, state in enumerate(states):
        if i >= num_data_points: break
        for j, other_state in enumerate(states):
            if j >= num_data_points: break
            interaction_frequencies[state][other_state].append(values[i][j])

    # Get mean and standard dev
    #means = np.array([[np.mean(interaction_frequencies[i][j]) for i in range(num_states)] for j in range(num_states)])
    stds = np.array([[np.std(interaction_frequencies[i][j]) for i in range(num_states)] for j in range(num_states)]).flatten()
    stds = stds[np.isfinite(stds)]
    #print(means, stds)
    return np.mean(stds)

if __name__ == '__main__':
    data = load_data("../data/GM12878_10k", "../data/loop_sequences_GM12878.fasta", "../data/epigenomic_tracks/GM12878.pickle")
    out_path = "../data/rnn_hyperparameters/" + sys.argv[1] + ".pickle"
    resolution = 100
    num_states = 2

    # Generate an X matrix for RNN training
    random_loops = [data[i] for i in random.sample(range(len(data)), 100)]
    del data
    X, ranges = generate_X(random_loops, spacing=resolution)

    Y_numerical = np.random.randint(0, num_states, size=X.shape[0])
    Y = np.zeros((X.shape[0], num_states))
    for i in range(Y_numerical.shape[0]):
        Y[i,Y_numerical[i]] = 1
    print(X.shape, Y.shape)

    consistencies = np.zeros((6, 6))
    for i, rec_nodes in enumerate([[5], [10], [25], [50], [25, 25], [50, 50]]):
        for j, dense_nodes in enumerate([[5], [10], [25], [50], [25, 25], [50, 50]]):
            print(rec_nodes, dense_nodes)
            model = rnn.RNNModel(recurrent_nodes=rec_nodes, dense_nodes=dense_nodes,
                                n_labels=num_states, n_features=X.shape[1], sequence_length=X.shape[2])
            model.create()
            model.train(X, Y, epochs=5)
            Y_pred = np.argmax(model.model.predict(X), axis=1)
            my_consistencies = [evaluate_consistency(random_loops[k][1], Y_pred[start:end], resolution, num_states) for k, (start, end) in enumerate(ranges)]
            consistencies[i, j] = sum(my_consistencies) / len(my_consistencies)
            print(i, j, consistencies[i, j])

    with open(out_path, 'wb') as file:
        pickle.dump(consistencies, file)
