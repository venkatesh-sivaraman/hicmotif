import sys
sys.path.insert(0, '/scratch/users/venkats/hic/straw/')

import numpy as np
import os
import straw
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import multiprocessing as mp
from functools import partial

FILE_SPLIT_COUNT = 300

def read_loops_worker(hic_file, resolution, min_size, max_size, row_info):
    i, row = row_info
    start = min(row['x1'], row['x2'])
    end = max(row['y1'], row['y2'])
    size = end - start
    if size < min_size or size > max_size:
        return i, None

    chr1_loc = ":".join([str(row['chr1']), str(start), str(end)])
    try:
        matrix = straw.straw("KR", hic_file, chr1_loc, chr1_loc, "BP", resolution)
        # Convert to list of triples instead of triple of lists
        triples = list(zip(*matrix))
        return i, (chr1_loc, triples)
    except:
        print("Exception")
        return i, None

def read_loops(hic_file, loops_file, out_prefix, resolution=25000, min_size=75000, max_size=300000):
    df = pd.read_csv(loops_file, delimiter='\t')
    file = None
    pool = mp.Pool(processes=6)
    worker = partial(read_loops_worker, hic_file, resolution, min_size, max_size)
    num_seen = 0
    for i, triples in pool.imap(worker, df.iterrows(), chunksize=10):
        if i % 10 == 0:
            print(i)
        if triples is None:
            continue
        if num_seen % FILE_SPLIT_COUNT == 0:
            if file is not None: file.close()
            print("Opening file", num_seen // FILE_SPLIT_COUNT)
            file = open(out_prefix + "_" + str(num_seen // FILE_SPLIT_COUNT) + ".pickle", "wb")
        pickle.dump(triples, file)
        num_seen += 1
    file.close()

task = int(sys.argv[1])
if task == 0:
    read_loops("../data/GSE63525_GM12878_insitu_primary+replicate_combined_30.hic",
               "../looplists/GSE63525_GM12878_primary+replicate_HiCCUPS_looplist_with_motifs.txt",
              "/scratch/users/venkats/hic/loops/GM12878_25k/loops", resolution=25000)
elif task == 1:
    read_loops("../data/GSE63525_GM12878_insitu_primary+replicate_combined_30.hic",
               "../looplists/GSE63525_GM12878_primary+replicate_HiCCUPS_looplist_with_motifs.txt",
              "/scratch/users/venkats/hic/loops/GM12878_10k/loops", resolution=10000)
elif task == 2:
    read_loops("../data/GSE63525_GM12878_insitu_primary+replicate_combined_30.hic",
               "../looplists/GSE63525_GM12878_primary+replicate_HiCCUPS_looplist_with_motifs.txt",
              "/scratch/users/venkats/hic/loops/GM12878_5k/loops", resolution=5000)
elif task == 3:
    read_loops("../data/GSE63525_IMR90_combined_30.hic",
               "../looplists/GSE63525_IMR90_HiCCUPS_looplist_with_motifs.txt",
              "/scratch/users/venkats/hic/loops/IMR90_25k/loops", resolution=25000)
elif task == 4:
    read_loops("../data/GSE63525_IMR90_combined_30.hic",
               "../looplists/GSE63525_IMR90_HiCCUPS_looplist_with_motifs.txt",
              "/scratch/users/venkats/hic/loops/IMR90_10k/loops", resolution=10000)
elif task == 5:
    read_loops("../data/GSE63525_IMR90_combined_30.hic",
               "../looplists/GSE63525_IMR90_HiCCUPS_looplist_with_motifs.txt",
              "/scratch/users/venkats/hic/loops/IMR90_5k/loops", resolution=5000)
