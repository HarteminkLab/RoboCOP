import numpy as np
import pysam
import math
import pandas
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects.packages import importr
import rpy2.robjects.vectors as vectors
import rpy2.robjects as ro
import io
from contextlib import redirect_stdout

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

    nucs = pandas.read_csv(nucFile, sep = '\t', header = None)
    nucs['dyad'] = (nucs[1] + nucs[2])/2
    nucs['dyad'] = nucs['dyad'].astype(int)
    nucs = nucs.rename(columns = {0: 'chr'})

    for i, r in nucs.iterrows():
        if "micron" in r['chr']: continue
        chrm = r['chr']
        if int(r['dyad']) - 73 - 15 > 0:
            segments.append({"chrm": chrm, "start": int(r['dyad']) - 73 - 15, "stop": int(r['dyad']) - 73})
            segments.append({"chrm": chrm, "start": int(r['dyad']) + 73, "stop": int(r['dyad']) + 73 + 15})
    return segments

def computeChrSegments(chrSizes):
    """
    Generate random segments 
    """
    segmentSize = 50
    np.random.seed(10)
    segments = []
    n = 5000
    chrs = list(chrSizes.keys())
    for i in range(n):
        chrm = chrs[np.random.choice(np.arange(len(chrs)))]
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
    f = io.StringIO()
    with redirect_stdout(f):
        params = fitdist.fitdist(vectors.IntVector(ec), 'nbinom', method = 'mle')
    params = params.rx2("estimate")
    mus = params.rx2("mu")[0]
    phis = params.rx2("size")[0]
    return mus, phis

def computeMNaseNucMusPhis(bamFile, nucleosomeFile, tmpDir, fragRange, offset = 0):
    empiricalCounts = []
    samfile = pysam.AlignmentFile(bamFile)

    nucs = pandas.read_csv(nucleosomeFile, sep = '\t', header = None)
    nucs['dyad'] = (nucs[1] + nucs[2])/2
    nucs['dyad'] = nucs['dyad'].astype(int)
    nucs = nucs.rename(columns = {0: 'chr'})

    for i1, r1 in nucs.iterrows():
        start = r1['dyad'] - 1 - 73
        stop = r1['dyad'] + 73
        chrm = r1['chr']
        
        counts = [0 for i in range(start, stop + 1)]
        # ignore chrXII
        if chrm == "chrXII": continue
        regions = samfile.fetch(chrm, max(0, start - 200), stop + 200)
        for r in regions:
            if r.template_length <= 0: continue
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
    fitdist = importr("fitdistrplus")
    f = io.StringIO()
    # nucleosomal DNA is 147 bases long
    for i in range(147):
        ec = empiricalCounts[:, i]
        with redirect_stdout(f):
            params = fitdist.fitdist(vectors.IntVector(ec), 'nbinom', method = 'mle')
        params = params.rx2("estimate")
        mus.append(params.rx2("mu")[0])
        phis.append(params.rx2("size")[0])
    return mus, phis


def computeMNaseBackground(bamFile, tmpDir, segments, fragRange, offset = 0, longShort = 'long'):
    """
    Compute MNase-seq midpoint of background distribution.
    """
    counts = []
    
    samfile = pysam.AlignmentFile(bamFile, "rb")
    for s in segments:
        minStart = s['start']
        maxEnd = s['stop'] + fragRange[1]

        region = samfile.fetch(s['chrm'], max(0, minStart - 200), maxEnd + 200)
        count = [0 for i in range(s['stop'] - s['start'])]
        for i in region:
            if i.template_length <= 0: continue
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
        f = io.StringIO()
        with redirect_stdout(f):
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

def computeMNaseBackgroundKernel(tmpDir, segments, signal):
    counts = []
    j = 0
    for s in segments:
        c = np.load(tmpDir + "/kernelized_counts_" + signal + "_"+ s["chrm"] + ".npy") 
        c = list(c[s['start'] : s['stop']])
        counts.extend(c)
    fitdist = importr('fitdistrplus')
    f = io.StringIO()
    with redirect_stdout(f):
        params = fitdist.fitdist(vectors.FloatVector(counts), 'gamma', method = "mme")
    params = params.rx2("estimate")
    shape = params.rx2("shape")[0]
    rate = params.rx2("rate")[0]
    params = {'shape': shape, 'rate': rate}
    return params

def computeMNaseTFPhisMus(bamFile, csvFile, tmpDir, fragRange, filename, offset = 0):
    """
    Negative binomial distribution for short fragments at TF
    binding sites.
    """
    tfCounts = []
    nucCounts = []
    samfile = pysam.AlignmentFile(bamFile, "rb")

    tfs = pandas.read_csv(csvFile, sep = '\t', header = None)
    tfs = tfs.rename(columns = {0: 'chr', 1: 'start', 2: 'end'})

    for i1, r1 in tfs.iterrows():
        mid = int(0.5*(r1['start'] + r1['end']))
        minStart = mid - 5
        maxEnd = mid + 5
        chrm = r1['chr']
        
        countMid = [0 for i in range(maxEnd - minStart + 1)]
        region = samfile.fetch(chrm, max(0, minStart - fragRange[1] - 1), maxEnd + fragRange[1] - 1)
        for i in region:
            if i.template_length <= 0: continue 
            if i.template_length - 2*offset >= 0:
                start = i.reference_start + 1 + offset
                end = i.reference_start + offset + i.template_length - 2*offset
            else:
                continue
            width = abs(i.template_length - 2*offset)
            if width >= fragRange[0] and width <= fragRange[1] and (start + end)/2 >= minStart and (start + end)/2 <= maxEnd: 
                countMid[int((start + end)/2 - minStart)] += 1
            
        tfCounts = tfCounts + countMid

    try:
        fitdist = importr('fitdistrplus')
        f = io.StringIO()
        with redirect_stdout(f):
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

def computeMNaseNucMusPhisKernel(tmpDir, nucleosomeFile, signal):
    empiricalCounts = []
    nucs = pandas.read_csv(nucleosomeFile, sep = '\t')
    nucs['dyad'] = nucs['smt_pos']
    # choose the 5k best nucleosomes
    nucs = nucs.sort_values(by = 'smt_value')[-5000:]

    for i1, r1 in nucs.iterrows():
        start = r1['dyad'] - 1 - 73
        stop = r1['dyad'] + 73
        chrm = r1['chr']

        counts = [0 for i in range(start, stop + 1)]
        # ignore chrXII
        if chrm == "chrXII": continue
        c = list(np.load(tmpDir + "/kernelized_counts_" + signal + "_" + chrm + ".npy")[start : stop])
        if np.sum(c) == 0: continue
        for i in range(len(c)): counts[i] = c[i]
        empiricalCounts.append(counts)

    empiricalCounts = np.array(empiricalCounts)
    means = []
    sds = []
    fitdist = importr("fitdistrplus")
    f = io.StringIO()
    for i in range(147):
        ec = empiricalCounts[:, i]
        with redirect_stdout(f):
            params = fitdist.fitdist(vectors.FloatVector(ec), 'gamma', method = 'mme')
        params = params.rx2("estimate")
        means.append(params.rx2("shape")[0])
        sds.append(params.rx2("rate")[0])
    return means, sds

def computeMNaseTFPhisMusKernel(tmpDir, csvFile, signal): 
    TFs = ["ABF1", "REB1"]
    counts = []
    j = 0 

    tfs = pandas.read_csv(csvFile, sep = '\t', header = None)
    tfs.columns = ['chr', 'start', 'end']

    if 1 == 1:
        for i1, r1 in tfs.iterrows():
            mid = int(0.5*(r1['start'] + r1['end']))
            minStart = mid - 5
            maxEnd = mid + 5
            chrm = r1['chr']
            
            c = np.load(tmpDir + "/kernelized_counts_" + signal + "_" + chrm + ".npy") 
            c = list(c[minStart - 1: maxEnd])
            if np.sum(c) == 0: continue
            counts.extend(c)
    fitdist = importr('fitdistrplus')
    f = io.StringIO()
    with redirect_stdout(f):
        params = fitdist.fitdist(vectors.FloatVector(counts), 'gamma') # , method = "mle")
    params = params.rx2("estimate")
    shape = params.rx2("shape")[0]
    rate = params.rx2("rate")[0]
    params = {'shape': shape, 'rate': rate}
    return params

