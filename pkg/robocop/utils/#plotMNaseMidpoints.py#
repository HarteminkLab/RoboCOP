# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pysam
import math
import pandas
import numpy as np
import sys
import matplotlib.gridspec as gridspec
import scipy.stats as stats
from scipy.interpolate import interpn


def plotMidpointsDensityAx(ax, MNaseFile, chrm, x_start, x_end, shortFragLim, longFragLim, offset = 0, ylims = (30, 220)):
    minStart = max(0, x_start - 200)
    maxEnd = x_end + 200
    count = np.zeros(maxEnd - minStart + 1)
    countMid = np.zeros(maxEnd - minStart + 1)
    countLong = np.zeros(maxEnd - minStart + 1)
    countMidLong = np.zeros(maxEnd - minStart + 1)
    samfile = pysam.AlignmentFile(MNaseFile, "rb")
    region = samfile.fetch(str(chrm), max(0, minStart - 500), maxEnd + 500)
    x = [] 
    y = [] 
    xy = np.zeros((maxEnd - minStart + 1, ylims[1] - ylims[0] + 1)) 
    for i in region:
        if abs(i.template_length - 2*offset) > ylims[1]: continue
        if i.template_length - 2*offset >= 0:
            start = i.reference_start + 1 + offset
            end = i.reference_start + offset + i.template_length - 2*offset
        else:
            continue
            end = i.reference_start + 1
            start = i.reference_start + i.template_length

        if (start + end)//2 < minStart or (start + end)//2 > maxEnd: continue
        width = abs(i.template_length - 2*offset)

        if width >= longFragLim[0] and width <= longFragLim[1]:
            if start >= minStart and end <= maxEnd:
                countLong[int(start-minStart):int(end-minStart)] += 1
                countMidLong[int((start + end)/2 - minStart)] += 1


        if width >= shortFragLim[0] and width <= shortFragLim[1]:
            if start >= minStart and end <= maxEnd:
                count[int(start-minStart):int(end-minStart)] += 1
                countMid[int((start + end)/2 - minStart)] += 1
        if width >= shortFragLim[0] and width <= shortFragLim[1]:
            x.append((start + end)//2)
            y.append(width - ylims[0])
            xy[(start + end)//2 - minStart, width - ylims[0]] += 1
        elif width >= longFragLim[0] and width <= longFragLim[1]:
            x.append((start + end)//2)
            y.append(width - ylims[0])
            xy[(start + end)//2 - minStart, width - ylims[0]] += 1
        else:
            x.append((start + end)//2)
            y.append(width - ylims[0])
            xy[(start + end)//2 - minStart, width - ylims[0]] += 1


    x = np.array(x)
    y = np.array(y)
    data , x_e, y_e = np.histogram2d(x, y, bins = 10, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    vmin = np.min(z)
    vmax = np.quantile(z, 0.9)
    
    idx_nuc = (y >= longFragLim[0]) & (y <= longFragLim[1])
    idx_short = (y <= shortFragLim[1])
    idx_else = ~idx_short & ~idx_nuc

    ax.scatter(x[idx_else], y[idx_else], c = z[idx_else], cmap = 'Greys', vmin = vmin, vmax = 2.5*vmax, s = 3)
    ax.scatter(x[idx_nuc], y[idx_nuc], c = z[idx_nuc], cmap = 'Reds', vmin = vmin, vmax = vmax, s = 3)
    ax.scatter(x[idx_short], y[idx_short], c = z[idx_short], cmap = 'Blues', vmin = vmin, vmax = 0.75*vmax, s = 3)
    ax.set_ylabel("Fragment length") #, fontsize = 15)
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(ylims)
    return countMid, countMidLong


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
        elif shortFragLim == (0, 0) and (start + end)/2 >= 61165 - 20 and int((start + end)/2) <= 61178 + 40 and width < 120:
            ax.plot(int((start + end)/2), width, 'go', alpha = 0.3, markersize = 3)
        elif shortFragLim == (0, 0) and int((start + end)/2) >= 62660 - 20 and int((start + end)/2) <= 62672 + 40 and width < 120:
            ax.plot(int((start + end)/2), width, 'go', alpha = 0.3, markersize = 3)
        else: ax.plot(int((start + end)/2), width, 'o', markeredgecolor = 'black', color = 'black', markeredgewidth=0.0, alpha = 0.3, markersize = 3)

    ax.set_ylabel("Fragment length") #, fontsize = 15)
    ax.set_xlim(minStart, maxEnd)
    if ylims == (0, 0): ax.set_ylim(30, 220)
    else: ax.set_ylim(ylims)
    return countMid, countMidLong

if __name__ == '__main__':
    MNaseFile = '/usr/xtmp/sneha/Chereji_2016/H2B_Input_MNase_200U.bam'
    chrm = 'chrI'
    start = 60500
    end = 65000
    shortFragLim = (0, 80)
    longFragLim = (127, 187)
    plotMidpointsAx(1, MNaseFile, chrm, start, end, shortFragLim, longFragLim)
