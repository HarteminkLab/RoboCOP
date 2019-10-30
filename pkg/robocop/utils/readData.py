##############################
# Read values form BAM file. #
##############################

import math
import roman
import numpy as np
import pysam

# Given MNase file, chromosome, start, and end, extract fragment midpoint counts of
# fragments of range given by fragRange
def getValuesMNaseOneFileFragmentRange(MNaseFile, chrm, minStart, maxEnd, fragRange):
    countMid = np.zeros(maxEnd - minStart + 1).astype(int)
    samfile = pysam.AlignmentFile(MNaseFile, "rb")
    region = samfile.fetch(str(chrm), max(0, minStart - fragRange[1] - 1), maxEnd + fragRange[1] - 1)
    for i in region:
        if i.template_length >= 0:
            start = i.reference_start + 1
            end = i.reference_start + i.template_length
        else:
            end = i.reference_start + 1
            start = i.reference_start + i.template_length
        width = abs(i.template_length)
        if width >= fragRange[0] and width <= fragRange[1] and (start + end)/2 >= minStart and (start + end)/2 <= maxEnd:
            countMid[int((start + end)/2 - minStart)] += 1
    return np.array(countMid)

def getValuesMNaseFragmentRange(bamFiles, chromosome, start, stop, fragRange):
    mnase = np.array([getValuesMNaseOneFileFragmentRange(x, chromosome, start, stop, fragRange) for x in bamFiles])
    return mnase

