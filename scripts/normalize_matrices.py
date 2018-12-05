import os, sys
import numpy as np
import statistics
import models
import pickle
from main import load_pickles

FILE_SPLIT_COUNT = 300

def generate_mats(directory):
    file_index = 0
    path = os.path.join(directory, "loops_{}.pickle".format(file_index))
    while os.path.exists(path):
        for id, data in load_pickles(path):
            try:
                yield models.InteractionMatrix(id, data)
            except ValueError:
                print("Omitting")
        file_index += 1
        path = os.path.join(directory, "loops_{}.pickle".format(file_index))

def collect_medians(mats):
    # Collect distributions of values over distances
    distances = {}
    for i, mat in enumerate(mats):
        if i % 100 == 0:
            print(i)
        for loc1, loc2 in mat.data:
            val = mat.data[(loc1, loc2)]
            distances.setdefault(abs(loc2 - loc1), []).append(val)

    # Calculate median for each distance
    print("Calculating medians of {} distances...".format(len(distances)))
    medians = {}
    for d, vals in distances.items():
        medians[d] = statistics.median(vals)
    print(medians)
    return medians

def normalize_matrices(mats, medians, out_dir):
    file = None
    for i, mat in enumerate(mats):
        if i % 100 == 0:
            print(i)
        if i % FILE_SPLIT_COUNT == 0:
            if file is not None: file.close()
            print("Opening file", i // FILE_SPLIT_COUNT)
            file = open(os.path.join(out_dir, "loops_" + str(i // FILE_SPLIT_COUNT) + ".pickle"), "wb")
        new_data = []
        for (x, y), f in mat.data.items():
            new_data.append((x, y, f / medians[abs(y - x)]))
        pickle.dump((mat.identifier, new_data), file)
    file.close()

if __name__ == '__main__':
    base_path = "/Users/venkatesh-sivaraman/Documents/School/MIT/6-047/proj/hicmotif/"
    input_dir = os.path.join(base_path, "data/GM12878_5k")
    output_dir = os.path.join(base_path, "data/normalized/GM12878_5k")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    medians = collect_medians(generate_mats(input_dir))
    print("Normalizing...")
    normalize_matrices(generate_mats(input_dir), medians, output_dir)
