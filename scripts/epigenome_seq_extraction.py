import pyBigWig
import os
import pandas as pd
import numpy as np
import pickle
import sys

epigenome_base = "/scratch/users/venkats/hic/data/histone_modifications"
looplists_path = "/scratch/users/venkats/hic/looplists"
output_base = "/scratch/users/venkats/hic/epigenomic_tracks"

#loading bigwig files from and converting into python objects

def gen_bigwig(epigenome_path):
    all_files = []
    for bb_file in os.listdir(epigenome_path):
        if bb_file[0] == '.': continue
        all_files.append((bb_file, pyBigWig.open(os.path.join(epigenome_path, bb_file))))
    all_files.sort()
    print("All files:", all_files)
    return all_files

if int(sys.argv[1]) == 1:
    epigenome_path = os.path.join(epigenome_base, "imr90")
    output_path = os.path.join(output_base, "IMR90.pickle")
    loop_file = os.path.join(looplists_path, "GSE63525_IMR90_HiCCUPS_looplist_with_motifs.txt")
else:
    epigenome_path = os.path.join(epigenome_base, "gm12878")
    output_path = os.path.join(output_base, "GM12878.pickle")
    loop_file = os.path.join(looplists_path, "GSE63525_GM12878_primary+replicate_HiCCUPS_looplist_with_motifs.txt")

list_bigwigs = gen_bigwig(epigenome_path)
loop_data = pd.read_table(loop_file)
# loop_data.head()

num_written = 0
file = open(output_path, 'wb')

for index, row in loop_data.iterrows():
    x = min(row["x1"], row["x2"])
    y = max(row["y1"], row["y2"])

    if index % 10 == 0:
        print(index)
    if y-x < 75000 or y-x > 300000:
        continue
    if row["chr1"] != row["chr2"]: print("not matching")

    chrom = "chr" + row["chr1"]
    entries = []

    for i in range(len(list_bigwigs)):
        marker_id, bb = list_bigwigs[i]

        #check bounds
        if y > bb.chroms(chrom):
            print("out of bound",index, marker_id, chrom, y, bb.chroms(chrom))
            continue

        peaks = bb.entries(chrom, x, y)
        if peaks == None: continue
        for p in peaks:
            start, stop, peak_data = p
            entries.append((i, max(x, start) - x, min(y, stop) - x))

    data = (row["chr1"]+":"+str(x)+":"+str(y), entries)
    pickle.dump(data, file)
    num_written += 1

file.close()
print("Finished writing {} entries to pickle.".format(num_written))
