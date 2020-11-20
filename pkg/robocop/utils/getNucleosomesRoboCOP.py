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
    # scores = [0 for j in range(chrSize)]
    scores = np.zeros(chrSize)
    # coords = coords[coords["chr"] == "chr" + roman.toRoman(chrm)]
    coords = coords[coords["chr"] == chrm]
    if coords.empty: 
        # print("HERE")
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
        # pTable = np.load(dirname + "posterior_table.idx" + str(i) + ".npy")
        pTable = np.load(dirname + "posterior_and_emission.idx" + str(i) + ".npz")['posterior']
        # nuc center is 73 bases away from nuc_start
        if i == 0:
                scores[start:end] = np.sum(pTable[:, nuc_center_start:nuc_center_end], axis = 1)
        elif i == idx[-1]:
            scores[start + 500 :end] = np.sum(pTable[500:, nuc_center_start:nuc_center_end], axis = 1)
        else:
            scores[start + 500 :end - 500] = np.sum(pTable[500:-500, nuc_center_start:nuc_center_end], axis = 1)
        del pTable
    return scores

def getNucScoresWrapper(lst):
    (coords, dirname, hmmconfig, c, chrSize) = lst
    scores = getNucScores(coords, dirname, hmmconfig, c, chrSize)
    return scores

def getNucs(dirname, chrSizes, hmmconfig):
    # Get the nucleosome dyad score for every position in the genome
    coordFile = dirname + "/coords.tsv"
    coords = pandas.read_csv(coordFile, sep = "\t")
    
    chrkeys = sorted(list(chrSizes.keys()))
    wrapperList = []
    for c in chrkeys:
        wrapperList.append((coords, dirname + "/tmpDir/", hmmconfig, c, chrSizes[c]))
    # scores = list(map(getNucScoresWrapper, wrapperList))
    scores = []
    for c in range(len(chrkeys)): scores.extend(getNucScores(coords, dirname + "/tmpDir/", hmmconfig, chrkeys[c], chrSizes[chrkeys[c]]))

    df = pandas.DataFrame(columns = ["chr", "dyad", "score"])
    chrs = sum(list(map(lambda x: [x for i in range(chrSizes[x])], chrkeys)), [])
    pos = sum(list(map(lambda x: [i + 1 for i in range(chrSizes[x])], chrkeys)), [])
    # scoresAll = sum(scores, [])
    df["chr"] = chrs
    df["dyad"] = pos
    # print(len(pos), len(scores))
    df["score"] = scores
    df.to_hdf(dirname + "/RoboCOP_outputs/nucCenterScores.h5", key = "df", mode = "w")
    #pickle.dump(score, open(dirname + "/RoboCOP_outputs/nucCenterScoresRoboCOP.p", "wb"))
    

# get nucleosome dyad predictions using a greedy approach from posterior decoding
def getNucPos(dirname, chrSizes):
    nucs = pandas.read_hdf(dirname + "/RoboCOP_outputs/nucCenterScores.h5", key = "df", mode = "r")
    idx = {}
    idxscores = {}
    chrkeys = sorted(list(chrSizes.keys()))
    for j in range(len(chrkeys)):
        idx[chrkeys[j]] = []
        idxscores[chrkeys[j]] = []
        nucchr = nucs[nucs["chr"] == chrkeys[j]]
        nucchr = nucchr.reset_index()
        while 1:
            i = np.argmax(nucchr["score"])
            print(i, j)
            if nucchr.iloc[i]["score"] <= 1e-1000: break
            idx[chrkeys[j]].append(i)
            idxscores[chrkeys[j]].append(nucchr.iloc[i]["score"])
            # replace surrounding scores with 0
            for k in range(max(0, i - 58), i + 59):
                nucchr.at[k, 'score'] = 0
            # nucchr.iloc[max(0, i - 58): i + 59, "score"] = 0
            
    # create data frame
    chrs = []
    dyads = []
    scores = []
    for i in chrSizes.keys():
        for j in range(len(idx[i])):
            chrs.append(i)
            dyads.append(int(idx[i][j]))
            scores.append(idxscores[i][j])
    a = pandas.DataFrame(columns = ["chr", "dyad", "score"])
    a["chr"] = chrs
    a["dyad"] = dyads
    a["score"] = scores
    a = a.sort_values(by = "score", ascending = False)
    print(a)
    a.to_hdf(dirname + "/RoboCOP_outputs/nucleosome_dyads.h5", key = "df", mode = "w")

if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print("Usage: python getNucleosomesRoboCOP.py <RoboCOP output dir>")
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
    if not os.path.exists(dirname + "/RoboCOP_outputs/nucCenterScores.h5"): getNucs(dirname, chrSizes, hmmconfig)

    # get nucleosome dyads using greedy approach
    # if not os.path.exists(dirname + "/RoboCOP_outputs/nucleosome_dyads.h5"): getNucPos(dirname, chrSizes)
    getNucPos(dirname, chrSizes)
    print("Done")
