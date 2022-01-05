import numpy as np
import pandas
import seaborn
import sys
from Bio import SeqIO
from configparser import SafeConfigParser
import pickle
import math
import pysam
import os
import re
from robocop.utils.getNucleosomesRoboCOP import getNucs, getNucPos

def match_nucs(dirname1, dirname2, hmmconfig1, hmmconfig2, chrSizes, min_prob = 0.1):
    df = pandas.DataFrame(columns = ['chr', 'dyadA', 'dyadB', 'occupancyA', 'occupancyB', 'shift'])
    if not os.path.isfile(dirname1 + "/RoboCOP_outputs/nucleosome_dyads.h5"):
        getNucs(dirname1, chrSizes, hmmconfig1)
        getNucPos(dirname1, chrSizes)
    if not os.path.isfile(dirname2 + "/RoboCOP_outputs/nucleosome_dyads.h5"):
        getNucs(dirname2, chrSizes, hmmconfig2)
        getNucPos(dirname2, chrSizes)

    nucs1 = pandas.read_hdf(dirname1 + "/RoboCOP_outputs/nucleosome_dyads.h5", key = 'df', mode = 'r')
    nucs2 = pandas.read_hdf(dirname2 + "/RoboCOP_outputs/nucleosome_dyads.h5", key = 'df', mode = 'r')
    for c in chrSizes:
        nucschr1 = nucs1[nucs1['chr'] == c]
        nucschr2 = nucs2[nucs2['chr'] == c]
        if nucschr1.empty and nucschr2.empty: continue
        nucschr1 = nucschr1[nucschr1['occ_score'] > min_prob]
        nucschr2 = nucschr2[nucschr2['occ_score'] > min_prob]

        nucschr1 = nucschr1.sort_values(by = 'dyad')
        nucschr2 = nucschr2.sort_values(by = 'dyad')
        nucschr1 = nucschr1.reset_index()
        nucschr2 = nucschr2.reset_index()

        print(nucschr1)
        print(nucschr2)
        l1 = len(nucschr1)
        l2 = len(nucschr2)

        i1 = 0
        i2 = 0

        max_nuc_dist = 73
        while i1 < l1 and i2 < l2:
            # print(i1, i2)
            r1 = nucschr1.iloc[i1]
            r2 = nucschr2.iloc[i2]
            if r1['dyad'] < r2['dyad']:
                if r2['dyad'] <= r1['dyad'] + max_nuc_dist:
                    df = df.append({"chr": c, "dyadA": r1['dyad'], "dyadB": r2['dyad'], 'occupancyA': r1['occ_score'], 'occupancyB': r2['occ_score'], 'shift': r2['dyad'] - r1['dyad']}, ignore_index = True)
                    i1 += 1
                    i2 += 1
                else:
                    df = df.append({"chr": c, "dyadA": r1['dyad'], "dyadB": None, 'occupancyA': r1['occ_score'], 'occupancyB': None, 'shift': None}, ignore_index = True)
                    i1 += 1
            else:
                if r1['dyad'] <= r2['dyad'] + max_nuc_dist:
                    df = df.append({"chr": c, "dyadA": r1['dyad'], "dyadB": r2['dyad'], 'occupancyA': r1['occ_score'], 'occupancyB': r2['occ_score'], 'shift': r2['dyad'] - r1['dyad']}, ignore_index = True)
                    i1 += 1
                    i2 += 1
                else:
                    df = df.append({"chr": c, "dyadA": None, "dyadB": r2['dyad'], 'occupancyA': None, 'occupancyB': r2['occ_score'], 'shift': None}, ignore_index = True)
                    i2 += 1

        for i in range(i1, l1):
            r1 = nucschr1.iloc[i1]
            df = df.append({"chr": c, "dyadA": r1['dyad'], "dyadB": None, 'occupancyA': r1['occ_score'], 'occupancyB': None, 'shift': None}, ignore_index = True)
        for i in range(i2, l2):
            r2 = nucschr2.iloc[i2]
            df = df.append({"chr": c, "dyadA": None, "dyadB": r2['dyad'], 'occupancyA': None, 'occupancyB': r2['occ_score'], 'shift': None}, ignore_index = True)

        # print(df)


    names = ["nuc_" + str(i) for i in range(len(df))]
    df['name'] = names
    df = df[['name', 'chr', 'dyadA', 'dyadB', 'occupancyA', 'occupancyB', 'shift']]
    return df

def nuc_map(dirname1, dirname2, outdir):
    configFile1 = dirname1 + "/config.ini"
    configFile2 = dirname2 + "/config.ini"
    outdir = outdir if outdir[-1] == '/' else outdir + '/' 
    os.makedirs(outdir, exist_ok = True)
    outfile = outdir + 'nuc_map.csv'
    
    config1 = SafeConfigParser()
    config1.read(configFile1)
    config2 = SafeConfigParser()
    config2.read(configFile2)
    
    # get size of each chromosome
    chrSizes = {}
    fastaFile = config1.get("main", "nucFile")
    fastaSeq = list(SeqIO.parse(open(fastaFile), 'fasta'))
    for fs in fastaSeq: chrSizes[fs.name] = len(fs.seq)

    bamFile1 = config1.get("main", "bamFile")
    bamFile2 = config2.get("main", "bamFile")
    hmmconfigfile1 = config1.get("main", "trainDir") + "/HMMconfig.pkl"
    hmmconfig1 = pickle.load(open(hmmconfigfile1, "rb"), encoding = "latin1")

    hmmconfigfile2 = config2.get("main", "trainDir") + "/HMMconfig.pkl"
    hmmconfig2 = pickle.load(open(hmmconfigfile2, "rb"), encoding = "latin1")

    if not os.path.isfile(outfile):
        nuc_df = match_nucs(dirname1, dirname2, hmmconfig1, hmmconfig2, chrSizes)
        nuc_df.to_csv(outfile, sep = '\t', index = False)
    else:
        nuc_df = pandas.read_csv(outfile, sep = '\t')
    return nuc_df

def get_nuc_categories(nuc_df, outdir):
    outfile = outdir + 'nuc_map_categories.csv'
    if os.path.isfile(outfile):
        nuc_df = pandas.read_csv(outfile, sep = '\t')
        return nuc_df
    
    categories = pandas.Series(['' for i in range(len(nuc_df))])
    dyads = np.zeros(len(nuc_df))
    
    categories[nuc_df['occupancyA'].isnull()] = 'null_A'
    categories[nuc_df['occupancyB'].isnull()] = 'null_B'
    categories[((nuc_df['occupancyA'].notnull()) & (nuc_df['occupancyB'].notnull()))] = 'not_null'

    dyads[nuc_df['occupancyA'].isnull()] = nuc_df[nuc_df['occupancyA'].isnull()]['dyadB']
    dyads[nuc_df['occupancyB'].isnull()] = nuc_df[nuc_df['occupancyB'].isnull()]['dyadA']
    dyads[((nuc_df['occupancyA'].notnull()) & (nuc_df['occupancyB'].notnull()))] = np.max(nuc_df[((nuc_df['occupancyA'].notnull()) & (nuc_df['occupancyB'].notnull()))][['dyadA', 'dyadB']], axis = 1)

    nuc_df['category'] = categories
    nuc_df['dyad'] = dyads.astype(int)

    nuc_df.to_csv(outfile, sep = '\t', index = False)
    return nuc_df
    
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python nuc_map_same_analysis.py <dirname1> <dirname2> <outdir>")
        exit(0)
    dirname1 = (sys.argv)[1]
    dirname2 = (sys.argv)[2]
    outdir = (sys.argv)[3]

    nuc_df = nuc_map(dirname1, dirname2, outdir)
    nuc_df = get_nuc_categories(nuc_df, outdir)
