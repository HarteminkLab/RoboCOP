# import matplotlib
# matplotlib.use('Agg')
from robocop.utils import visualization
from robocop import get_posterior_binding_probability_df
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import sys
import re
import pickle
import random
import glob
import h5py
import os
import configparser
from robocop.utils.plotMNaseMidpoints import plotMidpointsAx, plotMidpointsDensityAx
random.seed(9)

def get_idx(chrm, start, end, coords):
    coords = coords[(coords['chr'] == chrm) & (start <= coords['end']) & (end >= coords['start'])]
    return list(coords.index)

def calc_posterior(allinfofiles, dshared, coords, chrm, start, end):
    idxs = get_idx(chrm, start, end, coords)
    ptable = np.zeros((end - start + 1, dshared['n_states']))
    longCounts = np.zeros(end - start + 1)
    shortCounts = np.zeros(end - start + 1)
    counts = np.zeros(end - start + 1)
    for infofile in allinfofiles:
        for i in range(len(idxs)):
            idx = idxs[i]
            f = h5py.File(infofile, mode = 'r')
            k = 'segment_' + str(idx)
            if k not in f.keys():
                f.close()
                continue
            dshared['info_file'] = f
            dp = f[k + '/posterior'][:]
            lc = f[k + '/MNase_long'][:]
            sc = f[k + '/MNase_short'][:]
            f.close()
            dp_start = max(0, start - coords.loc[idx]['start'])
            dp_end = min(end - coords.loc[idx]['start'] + 1, coords.loc[idx]['end'] - coords.loc[idx]['start'] + 1)
            ptable_start = max(0, coords.loc[idx]['start'] - start)
            ptable_end = ptable_start + dp_end - dp_start
            ptable[ptable_start : ptable_end] += dp[dp_start : dp_end, :] 
            longCounts[ptable_start : ptable_end] += lc[dp_start : dp_end]
            shortCounts[ptable_start : ptable_end] += sc[dp_start : dp_end]
            counts[ptable_start : ptable_end] += 1 

    if counts[counts > 0].shape != counts.shape:
        print("ERROR: Invalid coordinates " + chrm + ":" + str(start) + "-" + str(end))
        print("Valid coordinates can be found at " + outDir + "coords.tsv")
        exit(0)
    ptable = ptable / counts[:, np.newaxis]
    longCounts = longCounts / counts
    shortCounts = shortCounts / counts
    optable = get_posterior_binding_probability_df(dshared, ptable)
    return optable, longCounts, shortCounts
    
def colorMap(outDir):
    if os.path.isfile(outDir + 'dbf_color_map.pkl'):
        dbf_color_map = pickle.load(open(outDir + "dbf_color_map.pkl", "rb"))
        return dbf_color_map

    print("Color map is not defined. Generating color map...")
    pwm = pickle.load(open(outDir + 'pwm.p', "rb"))
    predefined_dbfs = list(pwm.keys())
    # get upper case
    predefined_dbfs = [x for x in predefined_dbfs if x != "unknown"]
    predefined_dbfs = list(set([(x.split("_")[0]).upper() for x in predefined_dbfs]))
    n_tfs = len(predefined_dbfs)
    colorset48 = [(random.random(), random.random(), random.random(), 1.0) for i in range(n_tfs)] 
    nucleosome_color = '0.7'
    
    dbf_color_map = dict(list(zip(predefined_dbfs, colorset48)))
    dbf_color_map['nucleosome'] = nucleosome_color
    dbf_color_map['unknown'] =  '#D3D3D3'

    pickle.dump(dbf_color_map, open(outDir + "dbf_color_map.pkl", "wb"))
    print("Color map saved as", outDir + "dbf_color_map.pkl")
    return dbf_color_map


def plotRegion(gtffile, chrm, start, end, ax):
    a = pd.read_csv(gtffile, sep = "\t", header = None, comment = '#')
    a = a[(a[0] == chrm[3:]) & (a[3] <= end) & (a[4] >= start)]
    transcripts = {}
    # ax = plt.gca()
    for i, r in a.iterrows():
        if r[2] == 'transcript':
            if r[6] == '+':
                  ax.add_patch(patches.Rectangle((r[3], 0.4), r[4] - r[3] + 1, 0.3, color = 'skyblue'))
            else:
                  ax.add_patch(patches.Rectangle((r[3], -0.7), r[4] - r[3] + 1, 0.3, color = 'lightcoral'))
            gene_splits = dict([(g.split()[0], g.split()[1][1:-1]) for g in r[8][:-1].split(';')])
            gene = gene_splits['gene_name'] if 'gene_name' in gene_splits else gene_splits['gene_id']
            if gene not in transcripts:
                transcripts[gene] = (r[3], r[4], r[6])
            else:
                transcripts[gene] = (min(r[3], transcripts[gene][0]), max(r[4], transcripts[gene][1]), r[6])
        elif r[2] == 'exon': 
            if r[6] == '+':
                  ax.add_patch(patches.Rectangle((r[3], 0.1), r[4] - r[3] + 1, 0.9, color = 'skyblue'))
            else:
                  ax.add_patch(patches.Rectangle((r[3], -1), r[4] - r[3] + 1, 0.9, color = 'lightcoral'))
            gene_splits = dict([(g.split()[0], g.split()[1][1:-1]) for g in r[8][:-1].split(';')])
            gene = gene_splits['gene_name'] if 'gene_name' in gene_splits else gene_splits['gene_id']
        
    for t in transcripts:
        if transcripts[t][2] == '+':
            if transcripts[t][0] + 10 < start:
                ax.text(start, 1.2, t, fontsize = 12)
            else:
                ax.text(transcripts[t][0] + 10, 1.2, t, fontsize = 12)
        else:
            if transcripts[t][0] + 10 < start:
                ax.text(start, -2, t, fontsize = 12)
            else:
                ax.text(transcripts[t][0] + 10, -2, t, fontsize = 12)

    ax.set_xlim((start, end))
    ax.set_ylim((-1.9, 1.9))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

def plotOutputAx(outDir, config, dbf_color_map, optable, chrm, start, end, tech, longCounts, shortCounts, ax, save = True, gtffile = None):

    
    fragRangeLong = tuple([int(x) for x in re.findall(r'\d+', config.get("main", "fragRangeLong"))])
    fragRangeShort = tuple([int(x) for x in re.findall(r'\d+', config.get("main", "fragRangeShort"))])
    
    offset = 4 if tech == "ATAC" else 0
    if gtffile is not None:
        isgtf = 1
    else:
        isgtf = 0

    ax[1 + isgtf].plot(list(range(start - 1, end)), longCounts, color = 'maroon')
    ax[1 + isgtf].plot(list(range(start - 1, end)), shortCounts, color = 'blue')
    
    bamFile = config.get("main", "bamFile")
    shortCounts, longCounts = plotMidpointsDensityAx(ax[0 + isgtf], bamFile, chrm, start, end, fragRangeShort, fragRangeLong, offset = offset)
    # plot RoboCOP output
    visualization.plot_occupancy_profile(ax[2 + isgtf], op=optable, chromo=chrm, coordinate_start=start, threshold=0.1, dbf_color_map=dbf_color_map, plot_legend=False)
    ax[0 + isgtf].set_xlim((start, end))
    ax[1 + isgtf].set_xlim((start, end))
    ax[2 + isgtf].set_xlim((start, end))
    ax[0 + isgtf].set_xticks([])
    ax[1 + isgtf].set_xticks([])
    ax[2 + isgtf].set_xlabel(chrm)

    if isgtf: plotRegion(gtffile, chrm, start, end, ax[0])
    cols = list(set(optable.columns[optable.apply(max) >= 0.1]))
    return cols
    
def plot_output_ax(outDir, chrm, start, end, ax, isgtf, save = True):
    outDir = outDir + '/' if outDir[-1] != '/' else outDir

    configFile = outDir + "/config.ini"
    config = configparser.SafeConfigParser()
    config.read(configFile)
    
    # hmmconfigfile is generated only with robocop_em.py
    # if outputDir is generated using robocop_no_em.py
    # then use the dir path used to run robocop_no_em.py
    # to get the hmmconfig file
    
    hmmconfigfile = config.get("main", "trainDir") + "/HMMconfig.pkl"

    tech = config.get("main", "tech")

    gtfFile = config.get("main", "gtfFile")
    # create file for plotting
    dshared = pickle.load(open(hmmconfigfile, "rb"))

    if not isgtf: gtfFile = None
    
    coords = pd.read_csv(outDir + "coords.tsv", sep = "\t")
    allinfofiles = glob.glob(outDir + 'tmpDir/info*.h5')

    optable, longCounts, shortCounts = calc_posterior(allinfofiles, dshared, coords, chrm, start, end)
    dbf_color_map = colorMap(outDir) # pickle.load(open("dbf_color_map.pkl", "rb"))
    cols = plotOutputAx(outDir, config, dbf_color_map, optable, chrm, start, end, tech, longCounts, shortCounts, ax, save, gtffile = gtfFile)
    return cols
