import matplotlib
import pysam
import pandas
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# offset = 0 for MNase and offset = 4 for ATAC   
def getMNaseTSS(bamFile, minFrag, maxFrag, tss, offset=0):
    samfile = pysam.AlignmentFile(bamFile, 'rb')
    counts = np.zeros((200, 2001))
    seqCount = 0 # no. of sequences to compute avg. pileup
    for i, r in tss.iterrows():
        strand = '+' if r['ORF'].split('-')[0][-1] == 'W' else '-'
        if strand == '+':    
            start = int(r['coordinate']) - 1000
            end = int(r['coordinate']) + 1001
        else:
            start = int(r['coordinate']) - 1001
            end = int(r['coordinate']) + 1000
        region = samfile.fetch(r['chr'], start - 200, end + 200)
        seqCount += 1
        for j in region:
            if j.template_length - 2*offset > maxFrag or j.template_length - 2*offset < minFrag: continue
            rStart = j.reference_start + 1
            rEnd = rStart + j.template_length
            c = np.zeros(2001)
            mid = int(j.reference_start + offset + (j.template_length - 2*offset)/2)
            if mid < start or mid >= end: continue
            c[mid - int(start)] += 1
            if strand == '-': c = np.flip(c, 0)
            counts[j.template_length - 1] += c
    return counts, seqCount


def plotTSS(bamFile, tssFile, nucFrag, shortFrag, offset=0):
    minFrag = 45
    maxFrag = 200
    tss = pandas.read_csv(tssFile, sep='\t')
    tss = tss.dropna()

    counts, seqCount = getMNaseTSS(bamFile, minFrag, maxFrag, tss, offset) 
    
    counts = pandas.DataFrame(counts, columns = range(-1000, 1001))
    fig = plt.figure(figsize = (25, 20))

    gs = gridspec.GridSpec(2, 3, width_ratios = [0.5, 8, 3], height_ratios = [8, 3])
    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[1, 1])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])

    
    pileup_short = np.sum(counts[shortFrag[0]:shortFrag[1]], 0)
    pileup_short = pileup_short/float(seqCount)

    pileup_nuc = np.sum(counts[nucFrag[0]:nucFrag[1]], 0)
    pileup_nuc = pileup_nuc/float(seqCount)

    ax1.plot(range(-1000, 1001), pileup_short, color='blue')
    ax1.plot(range(-1000, 1001), pileup_nuc, color='darkred')
    ax1.set_xlabel('Position relative to TSS')
    im = seaborn.heatmap(counts, cmap = 'RdYlBu_r', ax = ax2, cbar_ax = ax0, xticklabels = 500, yticklabels = 50)
    ax2.invert_yaxis()
    ax2.set_ylim((minFrag, maxFrag))

    fragDist = np.sum(counts, 1)
    fragDist = fragDist/np.sum(fragDist)*100
    ax3.plot(fragDist, range(200), color='black')
    ax3.set_ylabel('Fragment length (bp)', fontsize = 18)
    ax3.set_ylim((minFrag, maxFrag))
    plt.suptitle('MNase-seq fragment midpoints', fontsize = 18)

    ax0.yaxis.set_ticks_position('left')
    ax0.yaxis.set_label_position('left')
    ax0.set_ylabel('Fragment midpoint counts', fontsize = 18)
    plt.savefig('./out.png')

    
