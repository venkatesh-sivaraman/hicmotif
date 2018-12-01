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

def state_asg(H, R, f):
    """ Returns the state assignment of the interaction matrix H when
    given H at resolution res, the state interaction score matrix R, and the scoring
    function
    """
    l = R.shape[0]
    res = H.resolution
    start, stop = H.range()
    n = (stop-start)//res
    scores = np.zeros((l,n))
    parent = np.zeros((l,n))

    for pos in range(1, n):
        for state_curr in range(l):
            cand_score = []
            for state_prev in range(l):
                score = 0 #scores[state_prev][pos-1]
                parent_state = state_prev
                for p in reversed(range(pos)):
                    score += f(state_curr, parent_state, np.log(1+H.value_at(pos*res+start, p*res+start)), R)
                    parent_state = int(parent[parent_state][p])
                cand_score.append(score)
            scores[state_curr][pos] = min(cand_score)
            parent[state_curr][pos] = int(np.argmin(np.array(cand_score)))


    state_curr = np.argmin(parent[:,-1])
    path = [state_curr]
    for p in reversed(range(1, n)):
        parent_state = int(parent[state_curr][p])
        state_curr = parent_state
        path.append(parent_state)
    return list(reversed(path))

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
