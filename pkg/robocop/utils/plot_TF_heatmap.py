import matplotlib
import pysam
import pandas
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# offset = 0 for MNase and offset = 4 for ATAC   
def getTFcounts(filename, tfFile, tf, offset=0):
    tfs = pandas.read_csv(tfFile, sep = "\t", header=None)
    tfs.columns = ['chr', 'start', 'end', 'TF', '0', 'strand']
    samfile = pysam.AlignmentFile(filename)
    tfs = tfs[tfs["TF"] == tf]
    counts = np.zeros((201, 1000))
    for i, r in tfs.iterrows():
        mid = int(r["start"] + r["end"])/2 - 1
        if mid < 500: continue
        start = mid - 500
        end = mid + 500
        regions = samfile.fetch(r["chr"], max(0, start - 200), end + 200)
        for region in regions:
            if region.template_length - 2*offset > 200: continue
            if region.template_length - 2*offset < 0: continue
            rMid = region.reference_start + (region.template_length + 2*offset)/2
            if rMid - start < 0 or rMid - start >= 1000: continue
            counts[int(region.template_length - 2*offset), int(rMid - start)] += 1
    return counts

def plotTF(bamFile, tfFile, nucFrag, shortFrag, tf, offset=0):
    tfCounts = getTFcounts(bamFile, tfFile, tf, offset)
    tfCounts = pandas.DataFrame(tfCounts[:, 250:750], columns = range(-250, 250))
    tfCounts = tfCounts[35:201]

    fig = plt.figure(figsize = (20, 20))

    gs = gridspec.GridSpec(2, 2, width_ratios = [0.5, 8], height_ratios = [8, 3])
    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[0, 1])
    ax2 = plt.subplot(gs[1, 1])
    seaborn.heatmap(tfCounts, ax = ax1, yticklabels = 15, cbar_ax = ax0, cmap = 'RdYlBu_r') # , vmax = 50
    ax1.set_xticks([])

    ax2.plot(range(-250, 250), np.sum(tfCounts.iloc[range(nucFrag[0]-35, nucFrag[1]-35)], axis = 0), color = 'darkred')
    ax2.plot(range(-250, 250), np.sum(tfCounts.iloc[range(shortFrag[1]-35)], axis = 0), color = 'blue')
    ax2.set_yticks([])
    ax2.set_xticks([-250, 0, 250])
    ax2.set_xticklabels([-250, 0, 250], fontsize = 12)
    ax2.set_xlim((-250, 250))
    ax2.set_xlabel("Distance from " + tf + " motif center", fontsize = 20)
    ax1.invert_yaxis()

