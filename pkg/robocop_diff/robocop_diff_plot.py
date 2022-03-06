import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import numpy as np
import pandas
import seaborn
import sys
from Bio import SeqIO
from configparser import SafeConfigParser
import glob
import pickle
import math
import pysam
import os
import re
from robocop_diff.multiple_nuc_alignment import nuc_map
from robocop_diff.nuc_diff_map import nuc_map_multiple
from robocop.utils.plotRoboCOP import calc_posterior, colorMap, plotRegion
from robocop.utils.plotMNaseMidpoints import plotMidpointsAx, plotMidpointsDensityAx

def get_info_robocop(outDir, chrm, start, end):
    outDir = outDir + '/' if outDir[-1] != '/' else outDir
    configFile = outDir + "/config.ini"
    config = SafeConfigParser()
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

    coords = pandas.read_csv(outDir + "coords.tsv", sep = "\t")
    allinfofiles = glob.glob(outDir + 'tmpDir/info*.h5')

    optable, longCounts, shortCounts = calc_posterior(allinfofiles, dshared, coords, chrm, start, end)
    dbf_color_map = colorMap(outDir) # pickle.load(open("dbf_color_map.pkl", "rb"))
    dbf_color_map['GCR1'] = 'blue'
    dbf_color_map['RAP1'] = 'red'
    
    info = {'config': config, 'dbf_color_map': dbf_color_map, 'optable': optable, 'tech': tech, 'longCounts': longCounts, 'shortCounts': shortCounts, 'gtffile': gtfFile}
    return info

def plot_1d(ax, infos, chrm, start, end):
    for d in range(len(ax)):
        ax[d].plot(list(range(start - 1, end)), infos[d]['longCounts'], color = 'maroon')
        ax[d].plot(list(range(start - 1, end)), infos[d]['shortCounts'], color = 'blue')
        if d == 0:
            maxx = max(np.max(infos[d]['longCounts']), np.max(infos[d]['shortCounts']))
        elif max(np.max(infos[d]['longCounts']), np.max(infos[d]['shortCounts'])) > maxx:
            maxx = max(np.max(infos[d]['longCounts']), np.max(infos[d]['shortCounts']))

    for d in range(len(ax)):
        ax[d].set_ylim((0, maxx))
        ax[d].set_xlim((start, end))
        ax[d].set_xticks([])
        ax[d].yaxis.set_label_position("right")
        ax[d].set_ylabel(infos[d]['tech'] + '_' + chr(ord('A') + d), fontsize=14)

def plot_2d(ax, infos, chrm, start, end):
    for d in range(len(ax)):
        bamFile = infos[d]['config'].get("main", "bamFile")
        fragRangeLong = tuple([int(x) for x in re.findall(r'\d+', infos[d]['config'].get("main", "fragRangeLong"))])
        fragRangeShort = tuple([int(x) for x in re.findall(r'\d+', infos[d]['config'].get("main", "fragRangeShort"))])
        offset = 4 if infos[d]['tech'] == "ATAC" else 0
        plotMidpointsDensityAx(ax[d], bamFile, chrm, start, end, fragRangeShort, fragRangeLong, offset = offset)
        ax[d].set_xlim((start, end))
        ax[d].set_xticks([])

def connect_m1_m2(rdyad, m1d_ax, m2d_ax, color, alpha):
    con1 = patches.ConnectionPatch(xyA=(rdyad - 25, m2d_ax.get_ylim()[0]), xyB=(rdyad - 25, m1d_ax.get_ylim()[1]), coordsA=m2d_ax.transData, coordsB=m1d_ax.transData, color=color, alpha=alpha, linewidth=0)
    m2d_ax.add_artist(con1)
    con2 = patches.ConnectionPatch(xyA=(rdyad + 26, m2d_ax.get_ylim()[0]), xyB=(rdyad + 26, m1d_ax.get_ylim()[1]), coordsA=m2d_ax.transData, coordsB=m1d_ax.transData, color=color, alpha=alpha, linewidth=0)
    m2d_ax.add_artist(con2)
    
    line2=con2.get_path().vertices
    line1=con1.get_path().vertices
    
    zoomcoords = sorted(np.concatenate((line1[[0, 2], :],line2[[0, 2], :])),key=lambda x: x[0])
    zoomcoords = [zoomcoords[1], zoomcoords[0]] + zoomcoords[2:]
    triangle = plt.Polygon(zoomcoords, linewidth=0, alpha=alpha, color=color, clip_on=False)
    m2d_ax.add_artist(triangle)

def connect_prev(rdyad, rdyadprevpos, m2d_ax, ax, color, alpha):
    con1 = patches.ConnectionPatch(xyA=(rdyadprevpos - 25, ax.get_ylim()[0]), xyB=(rdyad - 25, m2d_ax.get_ylim()[1]), coordsA=ax.transData, coordsB=m2d_ax.transData, color=color, alpha=alpha, linewidth=0)
    ax.add_artist(con1)
    con2 = patches.ConnectionPatch(xyA=(rdyadprevpos + 26, ax.get_ylim()[0]), xyB=(rdyad + 26, m2d_ax.get_ylim()[1]), coordsA=ax.transData, coordsB=m2d_ax.transData, color=color, alpha=alpha, linewidth=0)
    ax.add_artist(con2)
    line2=con2.get_path().vertices
    line1=con1.get_path().vertices

    zoomcoords = sorted(np.concatenate((line1[[0, 2], :],line2[[0, 2], :])),key=lambda x: x[0])
    zoomcoords = [zoomcoords[1], zoomcoords[0]] + zoomcoords[2:]
    triangle = plt.Polygon(zoomcoords, linewidth=0, alpha=alpha, color=color, clip_on=False)
    ax.add_artist(triangle)
    
def plot_nuc_dyad(ax, m2d_ax, m1d_ax, legend_ax, infos, nuc_df, chrm, start, end, ncol_nuc):
    issame = False
    posshift = False
    negshift = False
    for i in range(len(ax)):
        ax[i].plot(range(start-1, end), infos[i]['optable']['nuc_center'], color='grey')
    dyads = sorted(list(filter(lambda x: x.startswith('dyad'), list(nuc_df)))) # [:3]
    shifts = sorted(list(filter(lambda x: x.startswith('shift_'), list(nuc_df)))) # [:2]
    
    nuc_df['min_dyad'] = np.nanmin(nuc_df[dyads], axis=1)
    nuc_df['max_dyad'] = np.nanmax(nuc_df[dyads], axis=1)
    nucs = nuc_df[(nuc_df['chr'] == chrm) & (nuc_df['min_dyad'] + 26 >= start - 1) & (nuc_df['max_dyad'] - 25 <= end)]
    # exit(0)
    
    for d in range(len(dyads)):
        ax[d].set_xlim((start, end))
        ax[d].set_ylim((0, 1.15))
        ax[d].set_ylabel('P(DBF)')
        if d != len(dyads) - 1:
            ax[d].set_xticks([])
        else:
            ax[d].set_xlabel(chrm)
    
    for i, r in nucs.iterrows():
        curr_pos = 'A'
        for d in range(len(dyads)):
            dyad = 'dyad' + curr_pos
            if math.isnan(r[dyad]):
                prev_pos = curr_pos
                curr_pos = chr(ord(curr_pos) + 1)
                continue
            alpha = 0.9*r['pdyad' + curr_pos]
            if curr_pos == 'A':
                color = 'lightgrey'
                issame = True
                connectprev = False
            elif math.isnan(r['shift_' + prev_pos + curr_pos]):
                color = 'lightgrey'
                issame = True
                connectprev = False
            elif r['shift_' + prev_pos + curr_pos] < -20:
                color = 'orange'
                negshift = True
                connectprev = True
            elif r['shift_' + prev_pos + curr_pos]  > 20:
                color = 'green'
                posshift = True
                connectprev = True
            else:
                color = 'lightgrey'
                issame = True
                connectprev = True
            '''
            '''
            ax[d].add_patch(patches.Rectangle((r[dyad] - 25, 0), 51, 1.15, color=color, alpha=alpha, linewidth=0))
            m2d_ax[d].add_patch(patches.Rectangle((r[dyad] - 25, m2d_ax[d].get_ylim()[0]), 51, m2d_ax[d].get_ylim()[1] - m2d_ax[d].get_ylim()[0], color=color, alpha=alpha, linewidth=0))
            m1d_ax[d].add_patch(patches.Rectangle((r[dyad] - 25, 0), 51, m1d_ax[d].get_ylim()[1], color=color, alpha=alpha, linewidth=0))
            connect_m1_m2(r[dyad], m1d_ax[d], m2d_ax[d], color, alpha)
            connect_m1_m2(r[dyad], ax[d], m1d_ax[d], color, alpha)
            if connectprev: connect_prev(r[dyad], r['dyad' + prev_pos], m2d_ax[d], ax[d-1], color, alpha)
            '''
            '''
            prev_pos = curr_pos
            curr_pos = chr(ord(curr_pos) + 1)


    # ax[d].set_xticks([start + i*(end - start)//5 for i in range(5)] + [end])
    legend_ax.set_ylim((0, 2))
    legend_ax.set_xlim((start, end))
    legend_ax.axis('off')
    legend_ax_t = legend_ax.twinx()
    legend_ax_t.plot(0, 0, color='grey', label='nuc_dyad', linewidth=2)
    if issame: legend_ax_t.scatter(0, 0, s=150, marker='^', c='lightgrey', edgecolor='grey', alpha=0.8, label='no_shift_nuc')
    if negshift: legend_ax_t.scatter(0, 0, s=150, marker='<', c='orange', edgecolor='orange', alpha=0.6, label='neg_shift_nuc < -20')
    if issame: legend_ax_t.scatter(0, 0, s=150, marker='>', c='green', edgecolor='green', alpha=0.6, label='pos_shift_nuc > 20')
    legend_ax_t.set_ylim((0, 2))
    # legend_ax.set_xlim((start, end))
    legend_ax_t.legend(scatterpoints=1, frameon=False, ncol=ncol_nuc, prop={'size': 12}, loc=1)
    legend_ax_t.axis('off')
    
def plot_tf(ax, legend_ax, top_ax, infos, tf_df, chrm, start, end, ncol_tf):
    scores = sorted(list(filter(lambda x: x.startswith('score'), list(tf_df))))
    t_df = tf_df[(tf_df['chr'] == chrm) & (tf_df['start'] < end) & (tf_df['end'] >= start) & (np.any(tf_df[scores] > 0.1, axis=1))]
    if t_df.empty: return
    dbfs = sorted(list(set(t_df['TF'])))
    
    for i, r in t_df.iterrows():
        curr_pos = 'A'
        top_ax.plot((r['start'] + r['end'])//2 - 1, 240, marker='d', clip_on=False, color=infos[0]['dbf_color_map'][r['TF']], alpha=0.7)
        for d in range(len(ax)):
            ### ax[d].plot([(r['start'] + r['end'])//2, (r['start'] + r['end'])//2], [0, r['score' + curr_pos]], color=infos[d]['dbf_color_map'][r['TF']], linewidth=4, alpha=0.7)
            if d == 0:
                minprob = r['score' + curr_pos]
                maxprob = r['score' + curr_pos]
            else:
                if r['score' + curr_pos] < minprob:
                    minprob = r['score' + curr_pos]
                if r['score' + curr_pos] > maxprob:
                    maxprob = r['score' + curr_pos]

            curr_pos = chr(ord(curr_pos) + 1)

        # if ((maxprob - minprob > 0.3) or (minprob < 0.1 and maxprob > 0.1)):
        if (minprob < 0.1 and maxprob > 0.1):
            curr_pos = 'A'
            for d in range(len(ax)):
                ax[d].plot([(r['start'] + r['end'])//2, (r['start'] + r['end'])//2], [0, r['score' + curr_pos]], color=infos[d]['dbf_color_map'][r['TF']], linewidth=8, alpha=0.7)
                curr_pos = chr(ord(curr_pos) + 1)
            
            #### ax[d].plot((r['start'] + r['end'])//2 - 1, -0.05, marker='*', clip_on=False, color=infos[d]['dbf_color_map'][r['TF']])
            #### top_ax.plot((r['start'] + r['end'])//2 - 1, 240, marker='^', clip_on=False, color=infos[d]['dbf_color_map'][r['TF']], alpha=0.7)
            ### top_ax.plot((r['start'] + r['end'])//2 - 1, 240, marker='d', clip_on=False, color=infos[d]['dbf_color_map'][r['TF']], alpha=0.7)
            # curr_pos = 'A'
            # for d in range(len(ax)):
            #     ax[d].plot((r['start'] + r['end'])//2 - 1, r['score' + curr_pos] + 0.1, marker='*', clip_on=False, color=infos[d]['dbf_color_map'][r['TF']])
            #     curr_pos = chr(ord(curr_pos) + 1)
            
            
        '''
        tfvals = np.mean(infos[d]['optable'][list(filter(lambda x: x.split('_')[0].upper() == r['tf'], infos[d]['optable'].columns))], axis=1)[max(0, int(r['start']) - start) : min(int(r['end']) - start, len(infos[d]['optable']))]
        rstart = int(max(start, r['start']))
        rend = int(min(end + 1, r['end']))
        # print((start, end), (r['start'], r['end']), (rstart, rend))
        ax[d].plot(range(rstart, rend), tfvals, color=infos[d]['dbf_color_map'][r['tf']], alpha=0.5)
        ax[d].fill_between(range(rstart, rend), tfvals, color=infos[d]['dbf_color_map'][r['tf']], alpha=0.5)
        '''

    
    for dbf in dbfs:
        legend_ax.scatter(0, 0, s=50, marker='s', c=infos[0]['dbf_color_map'][dbf], edgecolor=infos[0]['dbf_color_map'][dbf], alpha=0.7, label=dbf)

    if dbfs != []:
        legend_ax.set_ylim((0, 2))
        legend_ax.set_xlim((start, end))
        legend_ax.legend(scatterpoints=1, frameon=False, ncol=ncol_tf, prop={'size': 12}, loc=2)
        
    legend_ax.axis('off')
    
def plot_diff_cop(dirname, dirnames, chrm, start, end, save=True, figsize=(19, 19), ncol_tf=4, ncol_nuc=2, filename=''):
    dirname = dirname + '/' if  dirname[-1] != '/' else dirname
    nuc_df = nuc_map_multiple(dirnames, dirname)
    # nuc_df = nuc_map(dirnames, dirname)
    tf_df = pandas.read_csv(dirname + 'diff_tf.csv', sep='\t')
    
    fig, ax = plt.subplots(len(dirnames)*4 + 3, 1, figsize=figsize, gridspec_kw={'height_ratios': [0.8, 0.2] + [1, 0.5, 1, 0.3] * len(dirnames) + [0.7]})
    plt.subplots_adjust(hspace = 0.01)
    
    infos = []
    for d in dirnames:
        infos.append(get_info_robocop(d, chrm, start, end))

    robocop_axes = [ax[4 * i + 2 + 2] for i in range(len(dirnames))]
    mnase2d_axes = [ax[4 * i + 0 + 2] for i in range(len(dirnames))]
    mnase1d_axes = [ax[4 * i + 1 + 2] for i in range(len(dirnames))]
    empty_axes = [ax[1]] + [ax[4 * i + 3 + 2] for i in range(len(dirnames))]
    legend_ax = ax[-1]

    for axs in empty_axes: axs.axis('off')
    plotRegion(infos[0]['gtffile'], chrm, start, end, ax[0])
    ax[0].axvline(117207, ls='--', color='black')
    ax[0].axvline(116645, ls='--', color='black')
    
    plot_1d(mnase1d_axes, infos, chrm, start, end)
    plot_2d(mnase2d_axes, infos, chrm, start, end)

    plot_tf(robocop_axes, legend_ax, mnase2d_axes[0], infos, tf_df, chrm, start, end, ncol_tf)
    plot_nuc_dyad(robocop_axes, mnase2d_axes, mnase1d_axes, legend_ax, infos, nuc_df, chrm, start, end, ncol_nuc)

    # plt.tight_layout()
    if save:
        os.makedirs(dirname + 'figures/', exist_ok = True)
        if filename == '':
            plt.savefig(dirname + "figures/robocop_diff_output_" + chrm + "_" + str(start) + "_" + str(end) + ".png")
            print("Output saved:", dirname + "figures/robocop_diff_output_" + chrm + "_" + str(start) + "_" + str(end) + ".png")
        else:
            plt.savefig(dirname + "figures/" + filename)
            print("Output saved:", dirname + "figures/" + filename)
            
