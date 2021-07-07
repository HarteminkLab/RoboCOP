import numpy as np
import pysam
# import roman
import math
import pandas
from rpy2.rinterface import RRuntimeError
from rpy2.robjects.packages import importr
import rpy2.robjects.vectors as vectors
import rpy2.robjects as ro

###############################################################################
## build nucleosome dinucleosome states emission by stacking 9 normal background state at the beginning, 128 single nucleotide states and 10 normal background state at the end
###############################################################################


nuc_emission = np.vstack((
[
[0.308512,0.191488,0.191488,0.308512],
[0.308512,0.191488,0.191488,0.308512], [0.308512,0.191488,0.191488,0.308512],
[0.308512,0.191488,0.191488,0.308512], [0.308512,0.191488,0.191488,0.308512],
[0.308512,0.191488,0.191488,0.308512], [0.308512,0.191488,0.191488,0.308512],
[0.308512,0.191488,0.191488,0.308512], [0.308512,0.191488,0.191488,0.308512]
],
[[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]] * 128,
[
[0.308512,0.191488,0.191488,0.308512], [0.308512,0.191488,0.191488,0.308512],
[0.308512,0.191488,0.191488,0.308512], [0.308512,0.191488,0.191488,0.308512],
[0.308512,0.191488,0.191488,0.308512], [0.308512,0.191488,0.191488,0.308512],
[0.308512,0.191488,0.191488,0.308512], [0.308512,0.191488,0.191488,0.308512],
[0.308512,0.191488,0.191488,0.308512], [0.308512,0.191488,0.191488,0.308512]
]))


##################NEED TO REMOVE#############################
def re_normalize(tf_prob, background_prob, nucleosome_prob):
    '''
    given a dict of tf_prob, a scaler of background_prob and
    a scaler of nucleosome_prob, re_normalize the numbers so that
    they sum to 1
    '''
    sum_prob = sum(tf_prob.values()) + background_prob + nucleosome_prob

    for tf in list(tf_prob.keys()):
        tf_prob[tf] /= sum_prob

    background_prob /= sum_prob
    nucleosome_prob /= sum_prob

    return tf_prob, background_prob, nucleosome_prob

def computeLinkers(nucFile):
    """
    get linker regions from file with nucleosome dyads
    """
    segments = []
    with open(nucFile) as infile:
        for line in infile:
            l = line.strip().split()
            if "micron" in l[0]: continue
            chrm = l[0]
            if int(l[1]) - 73 - 15 > 0:
                segments.append({"chrm": chrm, "start": int(l[1]) - 73 - 15, "stop": int(l[1]) - 73})
            segments.append({"chrm": chrm, "start": int(l[1]) + 73, "stop": int(l[1]) + 73 + 15})
    return segments

def computeChrSegments(chrSizes):
    """
    Generate random segments 
    """
    segmentSize = 50
    np.random.seed(10)
    segments = []
    n = 5000
    for i in range(n):
        chrm = np.random.randint(1, 17)
        start = np.random.randint(0, chrSizes[chrm] - segmentSize)
        segments.append({'chrm': chrm, 'start': start, 'stop': start + segmentSize})
    return segments

def computeMNaseNucOneMusPhis(bamFile, nucleosomeFile, fragRange):
    empiricalCounts = []
    samfile = pysam.AlignmentFile(bamFile)
    with open(nucleosomeFile) as infile:
        for line in infile:
            if line[0] == "#": continue
            l = line.strip().split()
            if l[0] == "chr": continue
            start = int(l[1]) - 73
            stop = int(l[1]) + 74
            counts = [0 for i in range(start, stop + 1)]
            # ignore chrXII
            if l[0] == "chrXII": continue
            regions = samfile.fetch(l[0], max(0, start - 200), stop + 200)
            for r in regions:
                if r.template_length <= 0: continue
                rStart = r.reference_start + 1
                rEnd = r.reference_start + r.template_length
                m = (rStart + rEnd)/2
                width = abs(r.template_length)
                if width < fragRange[0] or width > fragRange[1]: continue
                if m < start or m > stop: continue
                counts[int(m - start)] += 1
            empiricalCounts.extend(counts)
    empiricalCounts = np.array(empiricalCounts)
    mus = []
    phis = []
    ec = empiricalCounts
    fitdist = importr("fitdistrplus")
    params = fitdist.fitdist(vectors.IntVector(ec), 'nbinom', method = 'mle')
    params = params.rx2("estimate")
    mus = params.rx2("mu")[0]
    phis = params.rx2("size")[0]
    return mus, phis

def computeMNaseNucMusPhis(bamFile, nucleosomeFile, fragRange, offset = 0):
    empiricalCounts = []
    samfile = pysam.AlignmentFile(bamFile)
    with open(nucleosomeFile) as infile:
        for line in infile:
            if line[0] == "#": continue
            l = line.strip().split()
            if l[0] == "chr": continue
            start = int(l[1]) - 73
            stop = int(l[1]) + 74
            counts = [0 for i in range(start, stop + 1)]
            # ignore chrXII
            if l[0] == "chrXII": continue
            regions = samfile.fetch(l[0], max(0, start - 200), stop + 200)
            for r in regions:
                if r.template_length - 2*offset <= 0: continue
                rStart = r.reference_start + 1 + offset
                rEnd = r.reference_start + offset + r.template_length - 2*offset
                m = (rStart + rEnd)/2
                width = abs(r.template_length - 2*offset)
                if width < fragRange[0] or width > fragRange[1]: continue
                if m < start or m > stop: continue
                counts[int(m - start)] += 1
            empiricalCounts.append(counts)
    empiricalCounts = np.array(empiricalCounts)[:, :147]    
    mus = []
    phis = []
    # nucleosomal DNA is 147 bases long
    for i in range(147):
        ec = empiricalCounts[:, i]
        fitdist = importr("fitdistrplus")
        params = fitdist.fitdist(vectors.IntVector(ec), 'nbinom', method = 'mle')
        params = params.rx2("estimate")
        mus.append(params.rx2("mu")[0])
        phis.append(params.rx2("size")[0])
    return mus, phis

def computeMNaseBackground(bamFile, segments, fragRange, offset = 0):
    """
    Compute MNase-seq midpoint of background distribution.
    """
    counts = []
    samfile = pysam.AlignmentFile(bamFile, "rb")
    for s in segments:
        minStart = s['start']
        maxEnd = s['stop'] + fragRange[1]
        region = samfile.fetch(s['chrm'], minStart - 200, maxEnd + 200)
        count = [0 for i in range(s['stop'] - s['start'])]
        for i in region:
            if i.template_length - 2*offset <= 0: continue
            if i.template_length - 2*offset >= 0:
                start = i.reference_start + offset
                end = i.reference_start + offset + i.template_length - 1 - 2*offset
            else:
                continue
            width = abs(i.template_length - 2*offset)
            if width >= fragRange[0] and width <= fragRange[1] and (start + end)/2 >= s['start'] and (start + end)/2 < s['stop']:
                count[int((start + end)/2 - s['start'])] += 1
        counts = counts + count
    counts = np.array(counts)
    counts = counts.astype(int)
    try:
        fitdist = importr('fitdistrplus')
        params = fitdist.fitdist(vectors.IntVector(counts), 'nbinom', method = "mle")
        params = params.rx2("estimate")
        size = params.rx2("size")[0]
        mu = params.rx2("mu")[0]
        params = {'mu': mu, 'phi': size}
    except RRuntimeError:
        mu = 0.002
        phi = 100
        params = {'mu': mu, 'phi': phi}
    return params


def computeMNaseTFPhisMus(bamFile, csvFile, fragRange, filename, offset = 0):
    """
    Negative binomial distribution for short fragments at TF
    binding sites.
    """
    TFs = ["ABF1", "REB1"]
    tfCounts = []
    nucCounts = []
    samfile = pysam.AlignmentFile(bamFile, "rb")
    with open(csvFile) as infile:
        for line in infile:
            l = line.strip().split()
            if l[3] not in TFs: continue
            minStart = int(l[1])
            maxEnd = int(l[2])
            chrm = l[0]
            countMid = [0 for i in range(maxEnd - minStart + 1)]
            countNuc = [0 for i in range(maxEnd - minStart + 1)]
            region = samfile.fetch(chrm, minStart - fragRange[1] - 1, maxEnd + fragRange[1] - 1)
            for i in region:
                if i.template_length - 2*offset >= 0:
                    start = i.reference_start + 1 + offset
                    end = i.reference_start + offset + i.template_length - 2*offset
                else:
                    continue
                width = abs(i.template_length - 2*offset)
                if width >= fragRange[0] and width <= fragRange[1] and (start + end)/2 >= minStart and (start + end)/2 <= maxEnd: 
                    countMid[int((start + end)/2 - minStart)] += 1
            tfCounts = tfCounts + countMid
            nucCounts = nucCounts + countNuc
    try:
        fitdist = importr('fitdistrplus')
        p = fitdist.fitdist(vectors.IntVector(tfCounts), 'nbinom', method = "mle")
        p = p.rx2("estimate")
        size = p.rx2("size")[0]
        mu = p.rx2("mu")[0]
        params = {'mu': mu, 'phi': size}
    except Exception as e:
        # hard code values
        if e.args[0][:14] == "Error in (func":
            mu = 0.002
            phi = 100
            params = {'mu': mu, 'phi': phi}
    return params

