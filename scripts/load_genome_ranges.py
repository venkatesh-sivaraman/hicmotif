import sys
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import multiprocessing as mp
from functools import partial
from Bio import SeqIO
from Bio import Seq

# Speed up by not loading the same record twice consecutively
recent_record = None

def get_genome_subrecord(chromosome, loc1, loc2):
    """Creates a SeqRecord DNA sequence of the given chromosome from loc1 to loc2."""
    global recent_record
    if recent_record is None or recent_record.id != 'chr{}'.format(chromosome):
        recent_record = SeqIO.read("../data/hg19/chr{}.fa".format(chromosome), "fasta")

    id_str = ":".join(['chr{}'.format(chromosome), str(loc1), str(loc2)])
    new_seq = str(recent_record[loc1:loc2].seq).upper()
    return SeqIO.SeqRecord(Seq.Seq(new_seq), id=id_str)

def read_sequences_worker(min_size, max_size, row_info):
    i, row = row_info
    start = min(row['x1'], row['x2'])
    end = max(row['y1'], row['y2'])
    size = end - start
    if size < min_size or size > max_size:
        return None

    if i % 10 == 0:
        print(i)
    return get_genome_subrecord(row['chr1'], start, end)

def read_sequences(loops_file, out_file, min_size=75000, max_size=300000):
    df = pd.read_csv(loops_file, delimiter='\t')
    file = open(out_file, "w")
    #pool = mp.Pool(processes=6)
    worker = partial(read_sequences_worker, min_size, max_size)
    SeqIO.write((x for x in map(worker, df.iterrows()) if x is not None), file, "fasta")
    file.close()

if int(sys.argv[1]) == 0:
    read_sequences("../looplists/GSE63525_GM12878_primary+replicate_HiCCUPS_looplist_with_motifs.txt",
                   "/scratch/users/venkats/hic/loop_sequences_GM12878.fasta")
elif int(sys.argv[1]) == 1:
    read_sequences("../looplists/GSE63525_IMR90_HiCCUPS_looplist_with_motifs.txt",
                   "/scratch/users/venkats/hic/loop_sequences_IMR90.fasta")
else:
    print("Unknown argument.", sys.argv)
