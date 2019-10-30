from robocop.utils import visualization
from robocop import print_posterior_binding_probability
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import sys
import pickle
import random

def colorMap(pwmFile):
    pwm = pickle.load(open(pwmFile, "rb"), encoding = 'latin1')
    predefined_dbfs = list(pwm.keys())
    # get upper case
    predefined_dbfs = [x for x in predefined_dbfs if x != "unknown_TF"]
    predefined_dbfs = list(set([(x.split("_")[0]).upper() for x in predefined_dbfs]))
    n_tfs = len(predefined_dbfs)
    colorset48 = [(random.random(), random.random(), random.random(), 1.0) for i in range(n_tfs)] 
    nucleosome_color = '0.7'
    
    dbf_color_map = dict(list(zip(predefined_dbfs, colorset48)))
    dbf_color_map['nucleosome'] = nucleosome_color
    dbf_color_map['unknown_TF'] =  '#D3D3D3'
    return dbf_color_map

def plotOutput(filename, outDir, idx, pwmFile, dbf_color_map, chrm, start, end):
    fig, ax = plt.subplots(2, 1, sharex = True, figsize = (19, 7))
    # plot MNase-seq
    longCounts = np.load(outDir + "tmpDir/MNaseLong.idx" + str(idx) + ".npy")
    shortCounts = np.load(outDir + "tmpDir/MNaseShort.idx" + str(idx) + ".npy")
    ax[0].plot(list(range(start - 1, end)), longCounts, color = 'red')
    ax[0].plot(list(range(start - 1, end)), shortCounts, color = 'blue')
    opTable = pd.read_csv(filename, sep = '\t', low_memory = False)
    # plot RoboCOP output
    visualization.plot_occupancy_profile(ax[1], op = opTable, chromo = chrm, coordinate_start = start, pwmFile = pwmFile, threshold = 0.1, dbf_color_map = dbf_color_map)
    ax[1].set_xlim((start, end))
    plt.show()

if __name__ == '__main__':

    # hmmconfigfile is generated only with robocop_em.py
    # if outputDir is generated using robocop_no_em.py
    # then use the dir path used to run robocop_no_em.py
    # to get the hmmconfig file
    hmmconfigfile = "/Users/sneha/compBioBak/robocopTest/HMMconfig.pkl"
    outDir = "/Users/sneha/compBioBak/robocopTest/"
    pwmFile = "/Users/sneha/compBioBak/RoboCOP/analysis/motifs_meme.p"
    idx = 0 # index according to coordinate file
    chrm = 1
    start = 1
    end = 5000
    MNaseFile = "/Users/sneha/compBioBak/RoboCOP/analysis/dm504.bam"
    # create file for plotting
    dshared = pickle.load(open(hmmconfigfile, "rb"))
    d = pickle.load(open(outDir + "tmpDir/dict.idx" + str(idx) + ".pkl", "rb"))
    d['posterior_table'] = np.load(outDir + "tmpDir/posterior_table.idx" + str(idx) + ".npy")
    print_posterior_binding_probability(d, dshared, outDir + "em_test.out")
    filename = outDir + "em_test.out.timepoint" + str(idx)

    # create color map
    dbf_color_map = colorMap(pwmFile)

    plotOutput(filename, outDir, idx, pwmFile, dbf_color_map, chrm, start, end)
