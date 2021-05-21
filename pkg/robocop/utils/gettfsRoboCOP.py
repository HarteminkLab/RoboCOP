##################################################################################################
# Use the posterior decoding to compute TF binding sites in the genome.
##################################################################################################

import sys
import numpy as np
import pickle
import pandas
import os
from Bio import SeqIO
from configparser import SafeConfigParser

def getScores(coords, dirname, hmmconfig, chrm, tf, chrSize):
    scoresSum = list(map(lambda x: [0 for j in range(chrSize)], tf))
    coords = coords[coords["chr"] == chrm]
    if coords.empty: return scoresSum
    idx = list(coords.index)
    scores = []
    unknown = []
    otherTF = []
    nucs = []
    k = 0
    motifWidths = []
    for i in idx:
        # print(i)
        start = int(coords.iloc[k]["start"]) - 1
        end = int(coords.iloc[k]["end"])
        k += 1
        pTable = np.load(dirname + "posterior_and_emission.idx" + str(i) + ".npz")['posterior']
        for j in range(len(tf)):
            if i == 0:
                scoresSum[j][start:end] = list(np.sum(pTable[:, [hmmconfig["tf_starts"][tf[j]], hmmconfig["tf_starts"][tf[j]] + hmmconfig["tf_lens"][tf[j]]]], axis = 1))
            elif i == idx[-1]:
                scoresSum[j][start + 500 :end] = list(np.sum(pTable[500:, [hmmconfig["tf_starts"][tf[j]], hmmconfig["tf_starts"][tf[j]] + hmmconfig["tf_lens"][tf[j]]]], axis = 1))
            else:
                scoresSum[j][start + 500 :end - 500] = list(np.sum(pTable[500:-500, [hmmconfig["tf_starts"][tf[j]], hmmconfig["tf_starts"][tf[j]] + hmmconfig["tf_lens"][tf[j]]]], axis = 1))

    return scoresSum

def getScoresWrapper(args):
    (coords, dirname, hmmconfig, c, tfIndex, chrSize) = args
    scores = getScores(coords, dirname, hmmconfig, c, tfIndex, chrSize)
    return scores

def getMotifWidths(hmmconfig, tf):
    motifWidths = []
    for i in tf:
        motifWidths.append(hmmconfig["tf_lens"][i])
    return motifWidths


def getTFs(dirname, chrSizes, tfs, hmmconfig):
    coordFile = dirname + "/coords.tsv"
    coords = pandas.read_csv(coordFile, sep = "\t")

    chrkeys = sorted(list(chrSizes.keys()))
    for tf in tfs:
        tfIndex = list(filter(lambda x: (hmmconfig["tfs"][x].split("_")[0]).upper() == tf, range(len(hmmconfig["tfs"]))))
        motifWidths = getMotifWidths(hmmconfig, tfIndex)
        scoresPos = [list(map(lambda x: [0 for j in range(chrSizes[i])], tfIndex) for i in chrkeys)]
        scoresNeg = [list(map(lambda x: [0 for j in range(chrSizes[i])], tfIndex) for i in chrkeys)]
        wrapperList = []

        for c in chrkeys:
            wrapperList.append((coords, dirname + "/tmpDir/", hmmconfig, c, tfIndex, chrSizes[c]))
        scoresSum = [getScoresWrapper(x) for x in wrapperList]
        # compile scores
        totalLen = np.sum(list(chrSizes.values()))
        scores = np.zeros(totalLen)
        motifWidth = np.zeros(totalLen)
        chrs = sum(list(map(lambda x: [x for i in range(chrSizes[x])], chrkeys)), [])
        pos = sum(list(map(lambda x: [i + 1 for i in range(chrSizes[x])], chrkeys)), [])
        end = sum(list(map(lambda x: [i + 1 for i in range(chrSizes[x])], chrkeys)), [])
        i = 0
        for c in range(len(chrkeys)):
            for p in range(chrSizes[chrkeys[c]]):
                posMax = scoresSum[c][0][p]
                posIdx = 0
                for w in range(1, len(motifWidths)):
                    if scoresSum[c][w][p] > posMax:
                        posMax = scoresSum[c][w][p]
                        posIdx = w
                motifWidth[i] = motifWidths[posIdx]
                scores[i] = posMax
                end[i] = p + 1 + motifWidths[posIdx]
                if np.isnan(scores[i]) or np.isinf(scores[i]): scores[i] = 0
                i += 1
        df = pandas.DataFrame(columns = ["chr", "start", "end", "width", "score"])
        df["chr"] = chrs
        df["start"] = pos
        df["end"] = end
        df["width"] = motifWidth.astype(int)
        df["score"] = scores
        df.to_hdf(dirname + "/RoboCOP_outputs/" + tf + "_scores.h5", key = 'df', mode = 'w')
        print("Writing")

# filter out non zero and overlapping tf binding sites 
def getTFPosMod(dirname, chrSizes, tfs, hmmconfig):
    for tf in tfs:
        tfscores = pandas.read_hdf(dirname + "/RoboCOP_outputs/" + tf + "_scores.h5", key = "df", mode = "r")
        df = pandas.DataFrame(columns = list(tfscores))
        idxstart = {}
        idx = {}
        idxend = {}
        idxscores = {}
        chrkeys = sorted(list(chrSizes.keys()))
        for j in range(len(chrkeys)):
            idx[chrkeys[j]] = []
            idxscores[chrkeys[j]] = []
            tfchr = tfscores[tfscores['chr'] == chrkeys[j]]
            tfchrlen = len(tfchr)
            tfscore = np.array(tfchr['score'])
            tfchr = tfchr.reset_index()
            while 1: 
                i = np.argmax(tfscore)
                print(i, j, tfscore[i], file = sys.stderr)
                if tfscore[i] < 1e-1000: break
                # make that region 0
                mstart = i
                mend = tfchr.iloc[i]['end'] - 1
                mwidth = mend - mstart

                segstart = max(mstart - mwidth, 0)
                segend = min(mend + mwidth, tfchrlen)
                tfscore[segstart : segend] = 0
        
            tfchr['score'] = tfscore
            df = df.append(tfchr, ignore_index = True)

        df = df.sort_values(by = "score", ascending = False)
        df.to_hdf(dirname + "/RoboCOP_outputs/" + tf + ".h5", key = "df", mode = "w")

# filter out non zero and overlapping tf binding sites 
def getTFPos(dirname, chrSizes, tfs, hmmconfig):
    for tf in tfs:
        tfscores = pandas.read_hdf(dirname + "/RoboCOP_outputs/" + tf + "_scores.h5", key = "df", mode = "r")
        df = pandas.DataFrame(columns = list(tfscores))
        idxstart = {}
        idx = {}
        idxend = {}
        idxscores = {}
        chrkeys = sorted(list(chrSizes.keys()))
        for j in range(len(chrkeys)):
            idx[chrkeys[j]] = []
            idxscores[chrkeys[j]] = []
            tfchr = tfscores[tfscores['chr'] == chrkeys[j]]
            tfchr = tfchr.reset_index()
            while 1: 
                i = np.argmax(tfchr["score"])
                if tfchr.iloc[i]["score"] < 1e-100: break
                df = df.append(tfchr.iloc[i], ignore_index = True)
                # make that region 0
                mstart = tfchr.iloc[i]['start']
                mend = tfchr.iloc[i]['end']

                mask = ((tfchr['start'] <= mstart) & (tfchr['end'] >= mstart)) | ((tfchr['start'] <= mend) & (tfchr['end'] >= mend))
                tfchr.loc[mask, 'score'] = 0
        df = df.sort_values(by = "score", ascending = False)
        df.to_hdf(dirname + "/RoboCOP_outputs/" + tf + ".h5", key = "df", mode = "w")

                
if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print("Usage: python getNucleosomesRoboCOP.py <RoboCOP output dir> <TF -- optional>")
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

    os.makedirs(dirname + "/RoboCOP_outputs/", exist_ok = True)

    hmmconfigfile = config.get("main", "trainDir") + "/HMMconfig.pkl"
    hmmconfig = pickle.load(open(hmmconfigfile, "rb"), encoding = "latin1")
    tfmotifs = hmmconfig["tfs"]
    tfs = []
    for i in tfmotifs:
        tfs.append((i.split("_")[0]).upper())
    tfs = list(set(tfs))
    tfs = list(filter(lambda x: x != "UNKNOWN" and x != "BACKGROUND", tfs))
    tfs.sort()

    if len(sys.argv) == 3: tfIdx = int((sys.argv)[2])
    else: tfIdx = None

    if tfIdx: getTFs(dirname, chrSizes, [tfs[tfIdx]], hmmconfig)
    else: getTFs(dirname, chrSizes, tfs, hmmconfig)

