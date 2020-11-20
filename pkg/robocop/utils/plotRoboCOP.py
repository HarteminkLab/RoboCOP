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
import roman
import configparser
sys.path.insert(0, '/usr/project/compbio/sneha/scripts/')
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
    
    print(fragRangeLong)
    print(fragRangeShort)

    offset = 4 if tech == "atac" else 0
    opTable = pd.read_csv(filename, sep = '\t', low_memory = False) # [700:2350]
    fig, ax = plt.subplots(figsize = (4.4, 3.5))

    '''
    visualization.plot_occupancy_profile(ax, op = opTable, chromo = chrm, coordinate_start = start, pwmFile = pwmFile, threshold = 0.1, dbf_color_map = dbf_color_map)
    ax.set_xlim((start, end))
    ax.set_ylabel("P(DBF)")
    ax.set_yticks([0, 0.5, 1])
    plt.ylim((-0.005, 1.05))
    plt.show()
    '''
    fig, ax = plt.subplots(3, 1, sharex = True, figsize = (19, 7))
    # plot MNase-seq
    # longCounts = np.load(outDir + "tmpDir/MNaseLong.idx" + str(idx) + ".npy")
    # shortCounts = np.load(outDir + "tmpDir/MNaseShort.idx" + str(idx) + ".npy")
    # ax[0].plot(list(range(start - 1, end)), longCounts, color = 'red')
    # ax[0].plot(list(range(start - 1, end)), shortCounts, color = 'blue')
    
    shortCounts, longCounts = plotMidpointsAx(ax[0], MNaseFile, "chr" + roman.toRoman(chrm), start, end, fragRangeShort, fragRangeLong, offset = offset)
    ax[1].plot(range(start - 1, end), shortCounts, color = 'blue')
    ax[1].plot(range(start - 1, end), longCounts, color = 'maroon')
    opTable = pd.read_csv(filename, sep = '\t', low_memory = False) # [700:2350]
    # plot RoboCOP output
    # visualization.plot_occupancy_profile(ax[2], op = opTable, chromo = chrm, coordinate_start = start, pwmFile = pwmFile, threshold = 0.1, dbf_color_map = dbf_color_map)
    visualization.plot_occupancy_profile(ax[2], op = opTable, chromo = chrm, coordinate_start = start, pwmFile = pwmFile, threshold = 0.1, dbf_color_map = dbf_color_map)
    ax[1].set_xlim((start, end))
    ax[1].set_xlim((start, end))
    
    ax[0].set_xlim((start + 3000, end))
    ax[1].set_xlim((start + 3000, end))
    ax[2].set_xlim((start + 3000, end))
    # ax[2].set_xlim((start, end))
    # ax[1].set_xlim((start, end))
    # plt.show()
    # plt.savefig("/usr/xtmp/sneha/tmpDir/atactest.png")
    plt.savefig(outDir + "/atactest_" + str(idx) + ".png")
    
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
    # hmmconfigfile = "/usr/xtmp/sneha/robocop_DM504_new/HMMconfig.pkl"
    # hmmconfigfile = "/usr/xtmp/sneha/robocop_DM508_sacCer3/HMMconfig.pkl"
    # hmmconfigfile = "/usr/xtmp/sneha/robocop_os0_Chr4_sf0_80_nf127_187/HMMconfig.pkl"
    # hmmconfigfile = "/usr/xtmp/sneha/robocop_DM504_sacCer3_noUnknown/HMMconfig.pkl"
    # hmmconfigfile = "/usr/xtmp/sneha/robocop_no_unknown/HMMconfig.pkl"
    # hmmconfigfile = "/usr/xtmp/sneha/RoboCOP_par_check/HMMconfig.pkl"
    # outDir = "/usr/xtmp/sneha/robocopTestNew/"
    # outDir = "/usr/xtmp/sneha/robocop_DM504_1_16_new/"
    # outDir = "/usr/xtmp/sneha/robocop_DM504_new_heat_shock_0/"
    # outDir = "/usr/xtmp/sneha/robocop_DM508_new_heat_shock_60/"
    # outDir = "/usr/xtmp/sneha/robocop_DM508_new_heat_shock_60_new/"
    # outDir = "/usr/xtmp/sneha/robocop_no_threshold_no_unknown_Idx/"
    # outDir = "/usr/xtmp/sneha/compete_DM504_sacCer3_Test/"
    # outDir = "/usr/xtmp/sneha/robocop_atac_test_lin0_sf0_80_nf127_187/"
    # outDir = "/usr/xtmp/sneha/robocop_DM508_Test/"
    # outDir = "/usr/xtmp/sneha/compete_DM504_Test1/"
    # outDir = "/usr/xtmp/sneha/robocopTestCad_504/"
    # outDir = "/usr/xtmp/sneha/robocopTestCad/"
    # outDir = "/usr/xtmp/sneha/robocopTestCad_508/"
    # outDir = "/usr/xtmp/sneha/RoboCOP_par_check_Chr1_16/"
    # pwmFile = "/usr/xtmp/sneha/RoboCOP/analysis/motifs_meme.p"
    # pwmFile = "/usr/xtmp/sneha/RoboCOP/analysis/pwm_Gordan_motifs_withCXrap1.p"

    pwmFile = config.get("main", "pwmFile")
    tech = config.get("main", "tech")
    # idx = int(sys.argv)[2] # 95 # 619 # 0 # index according to coordinate file
    # chrm = 1 # 2 # 4 # 1
    # start = 60001 # 379453 # 1124001 # 60001
    # end = 65000 # 384453 # 1129000 # 65000
    # MNaseFile = "/usr/xtmp/sneha/data/MNase-seq/MacAlpine/DM504/dm504.bam" # "/Users/sneha/compBioBak/RoboCOP/analysis/dm504.bam"
    # MNaseFile = "/usr/xtmp/sneha/data/ATAC-seq/Schep/lin0/lin0_sacCer3.bam"
    MNaseFile = config.get("main", "bamFile")
    # create file for plotting
    dshared = pickle.load(open(hmmconfigfile, "rb"))

    coords = pd.read_csv(outDir + "coords.tsv", sep = "\t")

    if len(sys.argv) == 3:
        idxs = [int(sys.argv[2])]
    else:
        idxs = range(len(coords))
    # coords = pd.read_csv("/home/home3/sneha/Hartemink/MVCOMPETE/modifiedMVCOMPETE/pkg/unit_test/coord1.bed", sep = "\t")

    for idx in idxs:
        chrm = roman.fromRoman(coords.iloc[idx]["chr"][3:])
        start = int(coords.iloc[idx]["start"])
        end = int(coords.iloc[idx]["end"])
        d = pickle.load(open(outDir + "tmpDir/dict.idx" + str(idx) + ".pkl", "rb"))
        dp = np.load(outDir + "tmpDir/posterior_and_emission.idx" + str(idx) + ".npz")
        d['posterior_table'] = dp['posterior'] 
        print(np.shape(d['posterior_table']))
        # d['posterior_table'] = np.load(outDir + "tmpDir/posterior_table.idx" + str(idx) + ".npy")
        print_posterior_binding_probability(d, dshared, outDir + "em_test.out")
        filename = outDir + "em_test.out.timepoint" + str(idx)
        # exit(0)
        # create color map
        dbf_color_map = pickle.load(open("dbf_color_map.pkl", "rb"))
        # dbf_color_map = colorMap(pwmFile)
        plotOutput(filename, outDir, idx, pwmFile, dbf_color_map, chrm, start, end, tech)
        
