import matplotlib
import pysam
import pandas
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# offset = 0 for MNase and offset = 4 for ATAC   
def getNucCounts(filename, nucFile, offset = 0):
    nucs = pandas.read_csv(nucFile, sep = "\t", header=None)
    nucs.columns = ['chr', 'dyad', 'dyad+1']
    samfile = pysam.AlignmentFile(filename)
    counts = np.zeros((251, 200))
    for i, r in nucs.iterrows():
        mid = int(r["dyad"]) - 1
        if mid - 100 < 0: continue
        start = mid - 100
        end = mid + 100
        regions = samfile.fetch(r["chr"], max(0, start - 250), end + 250)
        for region in regions:
            if region.template_length - 2*offset > 250: continue
            if region.template_length - 2*offset < 0: continue
            rMid = region.reference_start + offset + (region.template_length - 2*offset)/2
            if rMid - start < 0 or rMid - start >= 200: continue
            counts[int(region.template_length - 2*offset), int(rMid - start)] += 1
    return counts


def plotNuc(bamFile, nucFile, nucFrag, shortFrag, offset=0):
    nucCounts = getNucCounts(bamFile, nucFile, offset)
    nucCounts = pandas.DataFrame(nucCounts, columns = range(-100, 100))
    nucCounts = nucCounts[35:201]

    fig = plt.figure(figsize = (20, 20))

    gs = gridspec.GridSpec(2, 2, width_ratios = [0.5, 8], height_ratios = [8, 3])
    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[0, 1])
    ax2 = plt.subplot(gs[1, 1])
    seaborn.heatmap(nucCounts, ax = ax1, yticklabels = 15, cbar_ax = ax0, cmap = 'RdYlBu_r') # vmax = 190, 
    ax1.set_xticks([])

    ax1.set_xticks([])
    ax2.plot(range(-100, 100), np.sum(nucCounts.iloc[range(nucFrag[0]-35, nucFrag[1]-35)], axis = 0), color = 'darkred')
    ax2.plot(range(-100, 100), np.sum(nucCounts.iloc[range(shortFrag[1]-35)], axis = 0), color = 'blue')
    ax2.set_yticks([])
    ax2.set_xticks([-100, 0, 100])
    ax2.set_xticklabels([-100, 0, 100], fontsize = 12)
    ax2.set_xlim((-100, 100))
    ax2.set_xlabel("Distance from dyad", fontsize = 20)
    ax1.invert_yaxis()

