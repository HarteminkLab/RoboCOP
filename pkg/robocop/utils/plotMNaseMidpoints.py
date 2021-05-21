import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pysam
import math
import pandas
import numpy as np
import sys
import matplotlib.gridspec as gridspec

def plotMidpointsAx(ax, MNaseFile, chrm, minStart, maxEnd, shortFragLim, longFragLim, offset = 0, ylims = (0, 0)):
    count = np.zeros(maxEnd - minStart + 1)
    countMid = np.zeros(maxEnd - minStart + 1)
    countLong = np.zeros(maxEnd - minStart + 1)
    countMidLong = np.zeros(maxEnd - minStart + 1)
    samfile = pysam.AlignmentFile(MNaseFile, "rb")
    region = samfile.fetch(str(chrm), max(0, minStart - 500), maxEnd + 500)
    for i in region:
        if abs(i.template_length - 2*offset) > 250: continue
        if i.template_length - 2*offset >= 0:
            start = i.reference_start + 1 + offset
            end = i.reference_start + offset + i.template_length - 2*offset
        else:
            continue
            end = i.reference_start + 1
            start = i.reference_start + i.template_length
        width = abs(i.template_length - 2*offset)

        if width >= longFragLim[0] and width <= longFragLim[1]:
            if start >= minStart and end <= maxEnd:
                countLong[int(start-minStart):int(end-minStart)] += 1
                countMidLong[int((start + end)/2 - minStart)] += 1


        if width >= shortFragLim[0] and width <= shortFragLim[1]:
            if start >= minStart and end <= maxEnd:
                count[int(start-minStart):int(end-minStart)] += 1
                countMid[int((start + end)/2 - minStart)] += 1
        if width >= shortFragLim[0] and width <= shortFragLim[1]: ax.plot(int((start + end)/2), width, color = 'blue', marker = 'o', markeredgecolor='blue', markeredgewidth=0.0, alpha = 0.3, markersize = 3)
        elif width >= longFragLim[0] and width <= longFragLim[1]: ax.plot(int((start + end)/2), width, 'o', color = 'darkred', markeredgecolor = 'darkred', alpha = 0.3, markersize = 3, markeredgewidth=0.0)
        else: ax.plot(int((start + end)/2), width, 'o', markeredgecolor = 'black', color = 'black', markeredgewidth=0.0, alpha = 0.3, markersize = 3)

    ax.set_ylabel("Fragment length")
    ax.set_xlim(minStart, maxEnd)
    if ylims == (0, 0): ax.set_ylim(30, 220)
    else: ax.set_ylim(ylims)
    return countMid, countMidLong
