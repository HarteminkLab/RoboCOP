##################################################################################################
# Use the posterior decoding to compute nucleosome dyad positions in the genome.
##################################################################################################

import sys
import numpy as np
import pickle
import pandas
import os
import glob
import h5py
from Bio import SeqIO
from scipy import sparse
# from configparser import SafeConfigParser
from configparser import ConfigParser

def get_sparse(f, k):
    g = f[k]
    v_sparse = sparse.csr_matrix((g['data'][:],g['indices'][:], g['indptr'][:]), g.attrs['shape'])
    return v_sparse

def get_sparse_todense(f, k):
    v_dense = np.array(get_sparse(f, k).todense())
    if v_dense.shape[0]==1: v_dense = v_dense[0]
    return v_dense


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


def getNucScores(coords, dirname, hmmconfig, r): 
    segmentSize = r['end'] - r['start'] + 1
    scores = np.zeros(segmentSize)
    scores_occ = np.zeros(segmentSize)
    scores_overlap = np.zeros(segmentSize)
    coords = coords[(coords["chr"] == r['chr']) & (coords['start'] >= r['start']) & (coords['end'] <= r['end'])]
    if coords.empty: 
        return [], []
    idx = list(coords.index)
    unknown = []
    otherTF = []
    nucs = []
    # k = 0
    motifWidths = []
    # 4 states for nuc_center
    nuc_center_start = hmmconfig["nuc_start"] + 9 + 4 + 4 * 63
    nuc_center_end = nuc_center_start + 4
    nuc_start = hmmconfig['nuc_start']
    nuc_end = nuc_start + hmmconfig['nuc_len']
    
    allinfofiles = glob.glob(dirname + 'info*.h5')

    for infofile in allinfofiles:
        f = h5py.File(infofile, mode = 'r')
        for i in idx:
            segment_key = 'segment_' + str(i)
            if segment_key not in f.keys(): continue
            start = int(coords.loc[i]["start"]) - r['start']
            end = int(coords.loc[i]["end"]) - r['start'] + 1
            # pTable = f[segment_key + '/posterior'][:]
            pTable = get_sparse_todense(f, segment_key + '/posterior')
            scores[start : end] += np.sum(pTable[:, nuc_center_start:nuc_center_end], axis = 1)
            scores_occ[start:end] += np.sum(pTable[:, nuc_start:nuc_end], axis = 1)
            scores_overlap[start : end] += 1
            del pTable
        f.close()
        
    # take average when multiple segments cover same genomic region 
    scores[scores_overlap > 0] /= scores_overlap[scores_overlap > 0]
    scores_occ[scores_overlap > 0] /= scores_overlap[scores_overlap > 0]
    # remove scores that were not calculated properly due to numerical issues
    scores[np.isnan(scores)] = 0
    scores_occ[np.isnan(scores_occ)] = 0
    scores[np.isinf(scores)] = 0
    scores_occ[np.isinf(scores_occ)] = 0
    # some scores are slightly > 1 due to numerical issues
    scores[scores > 1] = 1
    scores_occ[scores_occ > 1] = 1
    return scores, scores_occ

def getNucScoresWrapper(lst):
    (coords, dirname, hmmconfig, c, chrSize) = lst
    scores, scores_occ = getNucScores(coords, dirname, hmmconfig, c, chrSize)
    return list(scores), list(scores_occ)

def getNucs(dirname, chrSizes, hmmconfig):

    if os.path.isfile(dirname + "/RoboCOP_outputs/nucCenterScores.h5"): return
    
    os.makedirs(dirname + "/RoboCOP_outputs/", exist_ok = True)

    # Get the nucleosome dyad score for every position in the genome
    coordFile = dirname + "/coords.tsv"
    coords = pandas.read_csv(coordFile, sep = "\t")
    segments = getNonoverlappingSegments(coords)
    segments['width'] = segments['end'] - segments['start']
    scores = []
    scores_occ = []
    chrs = []
    pos = []
    segment_num = []
    for i, r in segments.iterrows(): # range(len(chrkeys)):
        s, so = getNucScores(coords, dirname + "/tmpDir/", hmmconfig, r)
        scores.extend(s)
        scores_occ.extend(so)
        chrs.extend([r['chr'] for _ in s])
        pos.extend([x + r['start'] for x in range(len(s))])
        segment_num.extend([i for _ in s])

    df = pandas.DataFrame(columns = ["chr", "dyad", "dyad_score", "occ_score", "segment_num"])
    df["chr"] = chrs
    df["dyad"] = pos
    df["dyad_score"] = scores
    df["occ_score"] = scores_occ
    df["segment_num"] = segment_num
    df.to_hdf(dirname + "/RoboCOP_outputs/nucCenterScores.h5", key = "df", mode = "w")
    

# get nucleosome dyad predictions using a greedy approach from posterior decoding
def getNucPos(dirname, chrSizes):

    if os.path.isfile(dirname + "/RoboCOP_outputs/nucleosome_dyads.h5"): return
    
    nucs = pandas.read_hdf(dirname + "/RoboCOP_outputs/nucCenterScores.h5", key = "df", mode = "r")
    idx = {}
    idxscores = {}
    idxscores_occ = {}
    idxscores_min_occ = {}
    segments = sorted(list(set(nucs['segment_num'])))
    segment_chrs = [nucs[nucs['segment_num'] == j].iloc[0]['chr'] for j in segments]
    for j in range(len(segments)):
        
        idx[segments[j]] = []
        idxscores[segments[j]] = []
        idxscores_occ[segments[j]] = []
        idxscores_min_occ[segments[j]] = []
        nucchr = nucs[nucs["segment_num"] == segments[j]]
        nucchr = nucchr.reset_index()
        nucchrlen = len(nucchr)
        arrchr = np.zeros(len(nucchr))
        print(nucchr)
        while len(nucchr[nucchr['dyad_score'] > 0]):
            i = np.argmax(nucchr["dyad_score"])
            # replace surrounding scores with 0
            nc = nucchr.iloc[i]['dyad_score']
            nc_occ = np.mean(nucchr.iloc[range(max(0, i - 58), min(nucchrlen, i + 59))]['occ_score'])
            nc_min_occ = np.min(nucchr.iloc[range(max(0, i - 58), min(nucchrlen, i + 59))]['occ_score'])
            idxcount = 0
            for k in range(max(0, i - 58), min(nucchrlen, i + 59)):
                if arrchr[k] == 0:
                    idxcount += 1
                    arrchr[k] = 1
                nucchr.at[k, 'dyad_score'] = 0
            if idxcount < 58 + 59: continue
            idx[segments[j]].append(nucchr.iloc[i]['dyad'])
            idxscores[segments[j]].append(nc)
            idxscores_occ[segments[j]].append(nc_occ)
            idxscores_min_occ[segments[j]].append(nc_min_occ)
            
    # create data frame
    chrs = []
    dyads = []
    scores = []
    scores_occ = []
    scores_min_occ = []
    for i in range(len(segments)): # chrSizes.keys():
        for j in range(len(idx[segments[i]])):
            chrs.append(segment_chrs[i])
            dyads.append(int(idx[segments[i]][j]))
            scores.append(idxscores[segments[i]][j])
            scores_occ.append(idxscores_occ[segments[i]][j])
            scores_min_occ.append(idxscores_min_occ[segments[i]][j])
    a = pandas.DataFrame(columns = ["chr", "dyad", "dyad_score", "avg_occ_score", "min_occ_score"])
    a["chr"] = chrs
    a["dyad"] = dyads
    a["dyad_score"] = scores
    a["avg_occ_score"] = scores_occ
    a["min_occ_score"] = scores_min_occ
    a.to_hdf(dirname + "/RoboCOP_outputs/nucleosome_dyads.h5", key = "df", mode = "w")

    
if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print("Usage: python getNucleosomesRoboCOP.py <RoboCOP output dir>")
        exit(0)
    
    dirname = (sys.argv)[1]
    configFile = dirname + "/config.ini"

    config = ConfigParser() # SafeConfigParser()
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
    getNucs(dirname, chrSizes, hmmconfig)
    # get nucleosome dyads using greedy approach
    getNucPos(dirname, chrSizes)
