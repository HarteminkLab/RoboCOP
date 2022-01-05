##################################################################################################
# Use the posterior decoding to compute TF binding sites in the genome.
##################################################################################################

import sys
import numpy as np
import pickle
import pandas
import h5py
import os
import glob
from Bio import SeqIO
from configparser import SafeConfigParser

# combine overlapping segments
def getNonoverlappingSegments(coords):
    chrs = list(set(coords['chr']))
    segments = pandas.DataFrame(columns = ['chr', 'start', 'end'])
    # get non overlapping segments for each chr
    for chrm in chrs:
        coords_chr = coords[coords['chr'] == chrm]
        # assuming each segment is of same length
        coords_chr = coords_chr.sort_values(by  = 'start')
        sstart = -1
        for i, r in coords_chr.iterrows():
            if sstart == -1:
                sstart = r['start']
                send = r['end']
            elif send + 1 <= r['end'] and send + 1 >= r['start']:
                send = r['end']
            else:
                segments = segments.append({'chr': chrm, 'start': sstart, 'end': send}, ignore_index = True)
                sstart = r['start']
                send = r['end']
        segments = segments.append({'chr': chrm, 'start': sstart, 'end': send}, ignore_index = True)
    return segments

def getScores(coords, dirname, hmmconfig, tf, r):
    chrm = r['chr']
    segmentSize = r['width'] + 1
    scoresSum = list(map(lambda x: [0 for j in range(segmentSize)], tf))
    scores_overlap = np.zeros(segmentSize)
    coords = coords[(coords["chr"] == r['chr']) & (coords['start'] >= r['start']) & (coords['end'] <= r['end'])]
    if coords.empty: 
        return []
    idx = list(coords.index)
    scores = []
    unknown = []
    otherTF = []
    nucs = []
    k = 0
    motifWidths = []

    allinfofiles = glob.glob(dirname + 'info*.h5')

    for infofile in allinfofiles:
        f = h5py.File(infofile, mode = 'r')
        for i in idx:
            segment_key = 'segment_' + str(i)
            if segment_key not in f.keys():
                continue
            start = int(coords.loc[i]["start"]) - r['start']
            end = int(coords.loc[i]["end"]) - r['start'] + 1
            pTable = f[segment_key + '/posterior'][:]
            for j in range(len(tf)):
                ss = np.sum(pTable[:, [int(hmmconfig["tf_starts"][int(tf[j])]), int(hmmconfig["tf_starts"][int(tf[j])] + hmmconfig["tf_lens"][int(tf[j])])]], axis = 1)
                ss[np.isinf(ss)] = 0
                ss[np.isnan(ss)] = 0
                scoresSum[j][start:end] = list(ss)
                scores_overlap[start : end] += 1
        f.close()

    for j in range(len(tf)):
        ss = np.array(scoresSum[j])
        ss[scores_overlap > 0] /= scores_overlap[scores_overlap > 0]
        ss[ss > 1] = 1
        scoresSum[j] = list(ss)
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

    os.makedirs(dirname + "/RoboCOP_outputs/", exist_ok = True)
    # Get the nucleosome dyad score for every position in the genome
    coordFile = dirname + "/coords.tsv"
    coords = pandas.read_csv(coordFile, sep = "\t")
    segments = getNonoverlappingSegments(coords)
    segments['width'] = (segments['end'] - segments['start']).astype(int)

    for tf in tfs:
        if os.path.isfile(dirname + "/RoboCOP_outputs/" + tf + "_scores.h5"):
            continue
        tfIndex = list(filter(lambda x: (hmmconfig["tfs"][x].split("_")[0]).upper() == tf, range(len(hmmconfig["tfs"]))))
        motifWidths = getMotifWidths(hmmconfig, tfIndex)

        scoresSum = []
        totalLen = 0
        chrs = []
        pos = []
        end = []

        for i, r in segments.iterrows():
            ss = getScores(coords, dirname + '/tmpDir/', hmmconfig, tfIndex, r)
            scoresSum.append(ss)
            totalLen += r['width']
            chrs.extend([r['chr'] for j in range(r['width'])])
            pos.extend([int(r['start']) + j for j in range(r['width'])])
            end.extend([int(r['start']) + j for j in range(r['width'])])
            
        # compile scores
        scores = np.zeros(totalLen)
        motifWidth = np.zeros(totalLen)

        i = 0
        for j, r in segments.iterrows():
            for p in range(r['width']):
                posMax = scoresSum[j][0][p]
                posIdx = 0
                for w in range(1, len(motifWidths)):
                    if scoresSum[j][w][p] > posMax:
                        posMax = scoresSum[j][w][p]
                        posIdx = w
                motifWidth[i] = motifWidths[posIdx]
                scores[i] = posMax
                end[i] = p + 1 + motifWidths[posIdx]
                if np.isnan(scores[i]) or np.isinf(scores[i]): scores[i] = 0
                i += 1
        df = pandas.DataFrame(columns = ["chr", "start", "end", "width", "score"])
        df["chr"] = chrs
        df["start"] = np.array(pos).astype(int)
        df["end"] = np.array(end).astype(int)
        df["width"] = motifWidth.astype(int)
        df["score"] = scores
        df.to_hdf(dirname + "/RoboCOP_outputs/" + tf + "_scores.h5", key = 'df', mode = 'w')
        print("Writing")

# filter out non zero and overlapping tf binding sites 
def getTFPosMod(dirname, chrSizes, tfs, hmmconfig):
    for tf in tfs:
        if os.path.isfile(dirname + "/RoboCOP_outputs/" + tf + ".h5"):
            continue
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
            if np.shape(tfscore)[0] == 0: continue
            tfscoreKeep = np.zeros(len(tfscore))
            tfchr = tfchr.reset_index()
            while 1:
                i = np.argmax(tfscore)
                print("Score:", i, j, tfscore[i], tfscore[i] < 1e-1000, np.isinf(np.log(tfscore[i])), file = sys.stderr)
                if tfscore[i] < 1e-1000 or np.isinf(np.log(tfscore[i])): break
                mstart = i
                mend = tfchr.iloc[i]['end'] - 1
                mwidth = mend - mstart

                segstart = max(mstart - mwidth, 0)
                segend = min(mend + mwidth, tfchrlen)
                tfsc = tfscore[i]
                tfscore[segstart : segend] = 0
                tfscoreKeep[(segstart + segend)//2] = tfsc
                
            tfchr['score'] = tfscoreKeep
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
                print("Score:", tfchr.iloc[i]["score"])
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

    print(tfs)
    print(len(tfs))
    print(tfs.index('HSF1'))
    if len(sys.argv) == 3: tfIdx = int((sys.argv)[2])
    else: tfIdx = None

    if tfIdx: getTFs(dirname, chrSizes, [tfs[tfIdx]], hmmconfig)
    else: getTFs(dirname, chrSizes, tfs, hmmconfig)

    if tfIdx: getTFPosMod(dirname, chrSizes, [tfs[tfIdx]], hmmconfig)
    else: getTFPosMod(dirname, chrSizes, tfs, hmmconfig)
