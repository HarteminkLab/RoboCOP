import matplotlib
matplotlib.use('Agg')
from robocop.utils import visualization
from robocop import print_posterior_binding_probability
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import sys
import re
import pickle
import random
import configparser
from plotMNaseMidpoints import plotMidpointsAx

def colorMap(pwmFile):
    pwm = pickle.load(open(pwmFile, "rb"), encoding = 'latin1')
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
    return dbf_color_map

def plotOutput(filename, outDir, idx, pwmFile, dbf_color_map, chrm, start, end, tech):
    
    fragRangeLong = tuple([int(x) for x in re.findall(r'\d+', config.get("main", "fragRangeLong"))])
    fragRangeShort = tuple([int(x) for x in re.findall(r'\d+', config.get("main", "fragRangeShort"))])
    
    offset = 4 if tech == "atac" else 0
    opTable = pd.read_csv(filename, sep = '\t', low_memory = False) # [700:2350]
    fig, ax = plt.subplots(figsize = (4.4, 3.5))

    fig, ax = plt.subplots(3, 1, sharex = True, figsize = (19, 7))
    # plot MNase-seq
    shortCounts, longCounts = plotMidpointsAx(ax[0], MNaseFile, chrm, start, end, fragRangeShort, fragRangeLong, offset = offset)
    ax[1].plot(range(start - 1, end), shortCounts, color = 'blue')
    ax[1].plot(range(start - 1, end), longCounts, color = 'maroon')
    opTable = pd.read_csv(filename, sep = '\t', low_memory = False) # [700:2350]
    # plot RoboCOP output
    visualization.plot_occupancy_profile(ax[2], op = opTable, chromo = chrm, coordinate_start = start, pwmFile = pwmFile, threshold = 0.1, dbf_color_map = dbf_color_map)
    ax[1].set_xlim((start, end))
    ax[1].set_xlim((start, end))
    
    ax[0].set_xlim((start + 3000, end))
    ax[1].set_xlim((start + 3000, end))
    ax[2].set_xlim((start + 3000, end))
    plt.savefig(outDir + "/robocop_output_" + str(idx) + ".png")
    
if __name__ == '__main__':

    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Usage: python plotRoboCOP.py outDir <idx -- optional>")
        exit(0)

    outDir = (sys.argv)[1]

    configFile = outDir + "/config.ini"
    config = configparser.SafeConfigParser()
    config.read(configFile)
    print(configFile)
    
    # hmmconfigfile is generated only with robocop_em.py
    # if outputDir is generated using robocop_no_em.py
    # then use the dir path used to run robocop_no_em.py
    # to get the hmmconfig file
    
    hmmconfigfile = config.get("main", "trainDir") + "/HMMconfig.pkl"
    pwmFile = config.get("main", "pwmFile")
    tech = config.get("main", "tech")
    MNaseFile = config.get("main", "bamFile")
    # create file for plotting
    dshared = pickle.load(open(hmmconfigfile, "rb"))

    coords = pd.read_csv(outDir + "coords.tsv", sep = "\t")

    if len(sys.argv) == 3:
        idxs = [int(sys.argv[2])]
    else:
        idxs = range(len(coords))

    for idx in idxs:
        chrm = coords.iloc[idx]["chr"]
        start = int(coords.iloc[idx]["start"])
        end = int(coords.iloc[idx]["end"])
        d = pickle.load(open(outDir + "tmpDir/dict.idx" + str(idx) + ".pkl", "rb"))
        dp = np.load(outDir + "tmpDir/posterior_and_emission.idx" + str(idx) + ".npz")
        d['posterior_table'] = dp['posterior'] 
        print_posterior_binding_probability(d, dshared, outDir + "em_test.out")
        filename = outDir + "em_test.out.timepoint" + str(idx)
        # create color map
        dbf_color_map = pickle.load(open("dbf_color_map.pkl", "rb"))
        plotOutput(filename, outDir, idx, pwmFile, dbf_color_map, chrm, start, end, tech)
        
