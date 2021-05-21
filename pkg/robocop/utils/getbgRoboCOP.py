##################################################################################################
# Use the posterior decoding to compute nucleosome dyad positions in the genome.
##################################################################################################

import sys
import numpy as np
import pickle
import pandas
import os
from Bio import SeqIO
from configparser import SafeConfigParser

def getNucScores(coords, dirname, hmmconfig, chrm, chrSize): 
    scores = np.zeros(chrSize)
    coords = coords[coords["chr"] == chrm]
    if coords.empty: 
        return scores
    idx = list(coords.index)
    unknown = []
    otherTF = []
    nucs = []
    k = 0
    motifWidths = []
    # 4 states for nuc_center
    nuc_center_start = hmmconfig["nuc_start"] + 9 + 4 + 4 * 63
    nuc_center_end = nuc_center_start + 4
    for i in idx:
        start = int(coords.iloc[k]["start"]) - 1
        end = int(coords.iloc[k]["end"])
        k += 1
        pTable = np.load(dirname + "posterior_and_emission.idx" + str(i) + ".npz")['posterior']
        # nuc center is 73 bases away from nuc_start
        if i == 0:
                scores[start:end] = pTable[:, 0]
        elif i == idx[-1]:
            scores[start + 500 :end] = pTable[500:, 0]
        else:
            scores[start + 500 :end - 500] = pTable[500:-500, 0]
        del pTable
    return scores

def getNucScoresWrapper(lst):
    (coords, dirname, hmmconfig, c, chrSize) = lst
    scores = getNucScores(coords, dirname, hmmconfig, c, chrSize)
    return scores

def getBG(dirname, chrSizes, hmmconfig):
    # Get the nucleosome dyad score for every position in the genome
    coordFile = dirname + "/coords.tsv"
    coords = pandas.read_csv(coordFile, sep = "\t")
    
    chrkeys = sorted(list(chrSizes.keys()))
    wrapperList = []
    for c in chrkeys:
        wrapperList.append((coords, dirname + "/tmpDir/", hmmconfig, c, chrSizes[c]))
    scores = []
    for c in range(len(chrkeys)): scores.extend(getNucScores(coords, dirname + "/tmpDir/", hmmconfig, chrkeys[c], chrSizes[chrkeys[c]]))

    df = pandas.DataFrame(columns = ["chr", "pos", "score"])
    chrs = sum(list(map(lambda x: [x for i in range(chrSizes[x])], chrkeys)), [])
    pos = sum(list(map(lambda x: [i + 1 for i in range(chrSizes[x])], chrkeys)), [])
    df["chr"] = chrs
    df["pos"] = pos
    df["score"] = scores
    df.to_hdf(dirname + "/RoboCOP_outputs/bgScores.h5", key = "df", mode = "w")
    

if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print("Usage: python getbgRoboCOP.py <RoboCOP output dir>")
        exit(0)
    
    dirname = (sys.argv)[1]
    configFile = dirname + "/config.ini"

    config = SafeConfigParser()
    config.read(configFile)
    
    # get size of each chromosome
    chrSizes = {}
    fastaFile = config.get("main", "nucFile")
    fastaSeq = list(SeqIO.parse(open(fastaFile), 'fasta'))
    for fs in fastaSeq: chrSizes[fs.name] = len(fs.seq)

    hmmconfigfile = config.get("main", "trainDir") + "/HMMconfig.pkl"
    hmmconfig = pickle.load(open(hmmconfigfile, "rb"), encoding = "latin1")

    os.makedirs(dirname + "/RoboCOP_outputs/", exist_ok = True)

    # first get the nucleosome dyad score for every position
    if not os.path.exists(dirname + "/RoboCOP_outputs/bgScores.h5"): getBG(dirname, chrSizes, hmmconfig)

