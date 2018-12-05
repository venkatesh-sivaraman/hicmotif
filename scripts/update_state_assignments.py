import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import models
import pickle
from Bio import SeqIO

def normalized_diff(s_i, s_j, H_ij, R):
    if H_ij == 0:
        return 0
    else:
        return abs(H_ij- R[s_i][s_j])/ H_ij

def penalty(s_dp, s_rnn):
    if s_dp != s_rnn:
        return 1.25
    else:
        return 1

def state_asg(H, R, f, penalty, res, rnn_states=None):
    """ Returns the state assignment of the interaction matrix H when
    given H at resolution res, the state interaction score matrix R, and the scoring
    function
    """
    H_vals = np.log(1+H.values_grid(res))
    l = R.shape[0]
    _, n = H.coordinates_grid(res)
    start = H.range()[0]
    scores = np.zeros((l,n))
    parent = np.zeros((l,n))

    store_change = []
    for pos in range(1, n):
        for state_curr in range(l):
            cand_score = []
            for state_prev in range(l):
                components_score = []
                score = scores[state_prev][pos-1]
                components_score.append(scores[state_prev][pos-1])
                parent_state = state_prev
                #score += f(state_curr, state_curr, H_vals[pos][pos], R)
                #components_score.append(f(state_curr, state_curr, H_vals[pos][pos], R))
                for p in reversed(range(pos)): #max(pos-100,0),
                    weight = ((pos-p)/pos) #(p/pos)**2 #
                    score += weight*f(state_curr, parent_state, H_vals[pos][p], R)
                    components_score.append(weight*f(state_curr, parent_state, H_vals[pos][p], R))
                    parent_state = int(parent[parent_state][p])
                if rnn_states is not None:
                    score *= penalty(state_curr, rnn_states[pos])
                cand_score.append(score)
                print("score components",components_score, " pos", pos, " state", (state_curr, state_prev))
                if pos == 187:
                    store_change.append(components_score)
            scores[state_curr][pos] = min(cand_score)
            parent[state_curr][pos] = int(np.argmin(np.array(cand_score)))


    state_curr = np.argmin(parent[:,-1])
    path = [state_curr]
    for p in reversed(range(1, n)):
        parent_state = int(parent[state_curr][p])
        state_curr = parent_state
        path.append(parent_state)
    return np.array(list(reversed(path))), store_change

def predicted_matrix_worker(S, R):
    """Returned matrix of predicted values given the state assignments S
    and the state interaction score matrix R"""
    p_matrix = np.zeros((len(S),len(S)))
    for i, s1 in enumerate(S):
        for j, s2 in enumerate(S):
            p_matrix[i][j] = R[s1][s2]
    return p_matrix

# def predicted_matrix(loop_file):
#     R = np.array([[7,2],[2,5]])
#     state_asg_matrices = []
#     for i in range(12):
#         identifier, test_item = pickle.load(file)
#         matrix = models.InteractionMatrix(identifier, test_item)
#         state_asg_matrices.append(state_asg(matrix, R, normalized_diff))
