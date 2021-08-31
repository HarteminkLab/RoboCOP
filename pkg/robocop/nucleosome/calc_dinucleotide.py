##############################################################################
# For given nucleosome file, calculate dinucleotide distribution
##############################################################################

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas
import seaborn
import math
import numpy as np
from Bio import SeqIO

def constructDinucFreq(sequences):
    dinucs = sum([[i + j for j in ['A', 'C', 'G', 'T']] for i in ['A', 'C', 'G', 'T']], [])
    df = pandas.DataFrame(columns = dinucs)
    counts = np.zeros((146, 16))
    
    for seq in sequences:
        d_idx = {}
        for d in dinucs:
            d_idx[d] = 0
        for i in range(len(seq) - 1):
            counts[i, dinucs.index(seq[i] + seq[i + 1])] += 1

    df = pandas.DataFrame(counts, columns = dinucs, index = range(-73, 73))
    df = df.div(df.sum(axis=1), axis=0)
    return df

def constructNucEmission(sequences):
    d_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    fst = np.zeros((9, 4))
    lst = np.zeros((10, 4))

    for i in range(len(sequences)):
        for s in range(9): fst[s, d_idx[sequences[i][s]]] += 1
        for s in range(10): lst[s, d_idx[sequences[i][s - 10]]] += 1

    fst = fst / np.sum(fst, axis = 1)[:, np.newaxis]
    lst = lst / np.sum(lst, axis = 1)[:, np.newaxis]

    fst = np.hstack((fst, np.zeros((9, 1))))
    lst = np.hstack((lst, np.zeros((10, 1))))
    
    nuc_emission = np.vstack((fst, [[1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0], [0, 0, 1.0, 0, 0], [0, 0, 0, 1.0, 0]] * 128, lst))
    return nuc_emission

def constructNucTransition(sequences):
    d_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    dinuc = np.zeros((127, 4, 4))
    strt = np.zeros(4)
    
    for i in range(len(sequences)):
        s_p = d_idx[sequences[i][9]]
        strt[s_p] += 1
        for s in range(10, 137):
            s_c = d_idx[sequences[i][s]]
            dinuc[s - 10, s_p, s_c] += 1
            s_p = s_c

    strt = strt / np.sum(strt) 
    for i in range(dinuc.shape[0]):
        dinuc[i] = dinuc[i] / np.sum(dinuc[i], axis = 1)[:, np.newaxis]
    return strt, dinuc

def read_dinuc_file(filename):
    dinuc = np.zeros((127, 4, 4))

    i = 0
    j = 0
    k = 0

    lcount = 0
    with open(filename) as infile:
        for line in infile:
            lcount += 1
            if lcount < 16: continue
            if lcount > 2047: break
            l = line.strip().split()
            dinuc[i, j, k] = float(l[2])

            k += 1
            if k == 4:
                k = 0
                j += 1

                if j == 4:
                    j = 0
                    i += 1

    return dinuc

def printDiNuc(strt, dinuc, filename):
    f = open(filename, 'w')
    f.write("# nucleosome dinucleotide model\n")
    f.write("# from to transition_prob\n")
    pos = 1
    while pos < 9:
        f.write(str(pos) + " " + str(pos + 1) + " " + str(1.0) + "\n")
        pos += 1

    f.write("# 1 branched background state\n")

    for i in range(4):
        f.write(str(pos) + " " + str(pos + i + 1) + " " + str(strt[i]) + "\n") 

    pos = pos + 1
    n_pos = pos + 4

    for i in range(dinuc.shape[0]):
        for j in range(pos, pos + 4):
            for k in range(n_pos, n_pos + 4):
                f.write(str(j) + " " + str(k) + " " + str(dinuc[i, j - pos, k - n_pos]) + "\n")

        n_pos += 4
        pos += 4 

    for i in range(4):
        f.write(str(pos + i) + " " + str(n_pos) + " " + str(1.0) + "\n")

    f.write("# 10 normal background states\n")
    pos += 4
    for i in range(9):
        f.write(str(pos) + " " + str(pos + 1) + " " + str(1.0) + "\n")
        pos += 1
    f.close()
        
def getDiNuc(nucleosomeFile, fastaFile, filename, genomeDinucFile = None):

    nucs = pandas.read_csv(nucleosomeFile, sep = '\t', header = None)
    nucs['dyad'] = (nucs[1] + nucs[2])/2
    nucs['dyad'] = nucs['dyad'].astype(int)
    nucs = nucs.rename(columns = {0: 'chr'})

    nucs = nucs.iloc[np.random.choice(len(nucs), size = 5000)]
    nucs = nucs.sort_values(by = 'chr')

    fastaSeq = list(SeqIO.parse(open(fastaFile), 'fasta'))
    sequences = []
    chrm = ''
    for i, r in nucs.iterrows():
        if chrm != r['chr']:
            chrm = r['chr']
            for fs in fastaSeq:
                if fs.name == chrm: chr_seq = str(fs.seq).upper()
                
        start = r['dyad'] - 1 - 73
        end = r['dyad'] + 73
        seq = chr_seq[start : end]
        if len(seq) < 147: continue
        if 'N' in seq: continue
        sequences.append(seq)

    nuc_emission = constructNucEmission(sequences)
    strt, dinuc = constructNucTransition(sequences)
    printDiNuc(strt, dinuc, filename)

    return nuc_emission

