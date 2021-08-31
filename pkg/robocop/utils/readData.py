##############################
# Read values form BAM file. #
##############################
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import numpy as np
import pysam
import pandas
import os
import sys

# Given MNase file, chromosome, start, and end, extract fragment midpoint counts of
# fragments of range given by fragRange

def getValuesMNaseOneFileFragmentRange(MNaseFile, chrm, minStart, maxEnd, fragRange, offset = 0):
    countMid = np.zeros(maxEnd - minStart + 1).astype(int)
    samfile = pysam.AlignmentFile(MNaseFile, "rb")
    region = samfile.fetch(chrm, max(0, minStart - fragRange[1] - 1), maxEnd + fragRange[1] - 1)
    for i in region:
        if i.template_length == 0: continue
        if i.template_length - 2*offset>= 0:
            start = i.reference_start + 1 + offset
            end = i.reference_start + offset + i.template_length - 2*offset
        else:
            # ignore reads on antisense strand
            continue
        width = abs(i.template_length)
        if width >= fragRange[0] and width <= fragRange[1] and (start + end)/2 >= minStart and (start + end)/2 <= maxEnd:
            countMid[int((start + end)/2 - minStart)] += 1

    return np.array(countMid)

def getValuesMNaseFragmentRange(bamFiles, chromosome, start, stop, fragRange):
    mnase = np.array([getValuesMNaseOneFileFragmentRange(x, chromosome, start, stop, fragRange) for x in bamFiles])
    return mnase

def getChrSizes(chrSizesFile):
    chrSizes = {}
    with open(chrSizesFile) as infile:
        for line in infile:
            l = line.strip().split()
            if l[0] == 'chrM': continue
            chrSizes[l[0]] = int(l[1])

    return chrSizes

def getMidpointCounts(samfile, c, chrSize, fragRange):
        counts = np.zeros((fragRange[1] - fragRange[0] + 1, chrSize)).astype(int)

        try:
            regions = samfile.fetch(c, 0, chrSize)
        except ValueError as ve:
            return None 
        
        for r in regions:
            if r.template_length <= 0: continue
            if r.template_length < fragRange[0]: continue
            if r.template_length > fragRange[1]: continue
            rStart = r.reference_start + 1 # + offset
            rEnd = r.reference_start + r.template_length # - 2*offset
            m = (rStart + rEnd)/2
            width = abs(r.template_length)
            counts[r.template_length - fragRange[0], int(m)] += 1

        return counts


def get2DValues(bamFile, chrSizesFile, fragRange, tmpDir):
    chrSizes = getChrSizes(chrSizesFile)
    samfile = pysam.AlignmentFile(bamFile)

    pop_c = ['chrM']
    
    if not os.path.isfile(tmpDir + "midpoint_counts.h5"):
        hdf = pandas.HDFStore(tmpDir + "midpoint_counts.h5", mode = "w")
        for c in chrSizes:
            if c == 'chrM': continue
            counts = getMidpointCounts(samfile, c, chrSizes[c], fragRange)
            if counts is None:
                pop_c.append(c)
                continue
            counts_df = pandas.DataFrame(counts.T, columns = range(counts.shape[0]))
            hdf.put(c, counts_df)
        hdf.close()

    hdf = pandas.HDFStore(tmpDir + "midpoint_counts.h5", mode = "r")

