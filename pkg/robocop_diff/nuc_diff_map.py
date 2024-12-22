import numpy as np
import pandas
import seaborn
import sys
from Bio import SeqIO
# from configparser import SafeConfigParser
from configparser import ConfigParser
import pickle
import math
import pysam
import os
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from robocop.utils.getNucleosomesRoboCOP import getNucs, getNucPos

def match_pair_nucs(dirname1, dirname2, hmmconfig1, hmmconfig2, chrSizes, min_prob = 0.1):
    df = pandas.DataFrame(columns = ['chr', 'dyadA', 'dyadB', 'occupancyA', 'occupancyB', 'shift_AB'])
    
    if not os.path.isfile(dirname1 + "/RoboCOP_outputs/nucleosome_dyads.h5"):
        getNucs(dirname1, chrSizes, hmmconfig1)
        getNucPos(dirname1, chrSizes)
    if not os.path.isfile(dirname2 + "/RoboCOP_outputs/nucleosome_dyads.h5"):
        getNucs(dirname2, chrSizes, hmmconfig2)
        getNucPos(dirname2, chrSizes)

    nucs1 = pandas.read_hdf(dirname1 + "/RoboCOP_outputs/nucleosome_dyads.h5", key = 'df', mode = 'r')
    nucs2 = pandas.read_hdf(dirname2 + "/RoboCOP_outputs/nucleosome_dyads.h5", key = 'df', mode = 'r')
    nucs1 = nucs1.dropna()
    nucs2 = nucs2.dropna()
    for c in chrSizes:
        nucschr1 = nucs1[nucs1['chr'] == c]
        nucschr2 = nucs2[nucs2['chr'] == c]
        if nucschr1.empty and nucschr2.empty: continue
        nucschr1 = nucschr1[nucschr1['min_occ_score'] > min_prob]
        nucschr2 = nucschr2[nucschr2['min_occ_score'] > min_prob]

        nucschr1 = nucschr1.sort_values(by = 'dyad')
        nucschr2 = nucschr2.sort_values(by = 'dyad')
        nucschr1 = nucschr1.reset_index(drop=True)
        nucschr2 = nucschr2.reset_index(drop=True)

        l1 = len(nucschr1)
        l2 = len(nucschr2)

        i1 = 0
        i2 = 0

        max_nuc_dist = 73
        while i1 < l1 and i2 < l2:
            r1 = nucschr1.iloc[i1]
            r2 = nucschr2.iloc[i2]
            if r1['dyad'] < r2['dyad']:
                if r2['dyad'] <= r1['dyad'] + max_nuc_dist:
                    df = df.append({"chr": c, "dyadA": r1['dyad'], "dyadB": r2['dyad'], 'occupancyA': r1['min_occ_score'], 'occupancyB': r2['min_occ_score'], 'pdyadA': r1['dyad_score'], 'pdyadB': r2['dyad_score'], 'shift_AB': r2['dyad'] - r1['dyad']}, ignore_index = True)
                    i1 += 1
                    i2 += 1
                else:
                    df = df.append({"chr": c, "dyadA": r1['dyad'], "dyadB": np.nan, 'occupancyA': r1['min_occ_score'], 'occupancyB': np.nan, 'pdyadA': r1['dyad_score'], 'pdyadB': np.nan, 'shift_AB': np.nan}, ignore_index = True)
                    i1 += 1
            else:
                if r1['dyad'] <= r2['dyad'] + max_nuc_dist:
                    df = df.append({"chr": c, "dyadA": r1['dyad'], "dyadB": r2['dyad'], 'occupancyA': r1['min_occ_score'], 'occupancyB': r2['min_occ_score'], 'pdyadA': r1['dyad_score'], 'pdyadB': r2['dyad_score'], 'shift_AB': r2['dyad'] - r1['dyad']}, ignore_index = True)
                    i1 += 1
                    i2 += 1
                else:
                    df = df.append({"chr": c, "dyadA": np.nan, "dyadB": r2['dyad'], 'occupancyA': np.nan, 'occupancyB': r2['min_occ_score'], 'pdyadA': np.nan, 'pdyadB': r2['dyad_score'], 'shift_AB': np.nan}, ignore_index = True)
                    i2 += 1

        for i in range(i1, l1):
            r1 = nucschr1.iloc[i]
            df = df.append({"chr": c, "dyadA": r1['dyad'], "dyadB": np.nan, 'occupancyA': r1['min_occ_score'], 'occupancyB': np.nan, 'pdyadA': r1['dyad_score'], 'pdyadB': np.nan, 'shift_AB': np.nan}, ignore_index = True)
        for i in range(i2, l2):
            r2 = nucschr2.iloc[i]
            df = df.append({"chr": c, "dyadA": np.nan, "dyadB": r2['dyad'], 'occupancyA': np.nan, 'occupancyB': r2['min_occ_score'], 'pdyadA': np.nan, 'pdyadB': r2['dyad_score'], 'shift_AB': np.nan}, ignore_index = True)

        # print(df)


    names = ["nuc_" + str(i) for i in range(len(df))]
    df['name'] = names
    df = df[['name', 'chr', 'dyadA', 'dyadB', 'occupancyA', 'occupancyB', 'pdyadA', 'pdyadB', 'shift_AB']]
    return df

def match_new_nucs(nuc_df, dirname, hmmconfig, chrSizes, min_prob = 0.1):
    if not os.path.isfile(dirname + "/RoboCOP_outputs/nucleosome_dyads.h5"):
        getNucs(dirname, chrSizes, hmmconfig)
        getNucPos(dirname, chrSizes)

    nucs = pandas.read_hdf(dirname + "/RoboCOP_outputs/nucleosome_dyads.h5", key = 'df', mode = 'r')
    nucs = nucs.dropna()

    dyads = sorted(list(filter(lambda x: x[:-1] == 'dyad', list(nuc_df))))

    curr_idx = dyads[-1][4:]
    next_idx = chr(ord(curr_idx) + 1)
    df = pandas.DataFrame(columns = list(nuc_df) + ['dyad' + next_idx, 'occupancy' + next_idx, 'shift_' + curr_idx + next_idx])

    nuc_df['min_dyad'] = np.nanmin(np.array(nuc_df[dyads].values), axis=1).astype(int)
    nuc_df['max_dyad'] = np.nanmax(np.array(nuc_df[dyads].values), axis=1).astype(int)
    nuc_df['mean_dyad'] = np.nanmean(np.array(nuc_df[dyads].values), axis=1).astype(int)
    nuc_df['curr_dyad'] = nuc_df[dyads[-1]]
    nuc_df = nuc_df.sort_values(by=['chr', 'mean_dyad'] + dyads)
    nuc_df = nuc_df.reset_index(drop=True)

    for c in chrSizes:
        nucschr = nucs[nucs['chr'] == c]
        nuc_df_chr = nuc_df[nuc_df['chr'] == c]
        if nucschr.empty and nuc_df_chr.empty: continue

        # nuc_df_chr = nuc_df_chr.sort_values(by=dyads)
        # nuc_df_chr = nuc_df_chr.reset_index(drop=True)
        
        nucschr = nucschr[nucschr['min_occ_score'] > min_prob]
        nucschr = nucschr.sort_values(by = 'dyad')
        nucschr = nucschr.reset_index()

        l1 = len(nuc_df_chr)
        l2 = len(nucschr)

        i1 = 0
        i2 = 0

        max_nuc_dist = 73
        while i1 < l1 and i2 < l2:
            r1 = nuc_df_chr.iloc[i1]
            r2 = nucschr.iloc[i2]
            if math.isnan(r1['curr_dyad']):
                if r1['min_dyad'] - max_nuc_dist <= r2['dyad'] and r1['max_dyad'] + max_nuc_dist >= r2['dyad']:
                    r_loc = r1.to_dict()
                    r_loc['dyad' + next_idx] = r2['dyad']
                    r_loc['occupancy' + next_idx] = r2['min_occ_score']
                    r_loc['pdyad' + next_idx] = r2['dyad_score']
                    r_loc['shift_' + curr_idx + next_idx] = np.nan
                    df = df.append(r_loc, ignore_index=True) 
                    i1 += 1
                    i2 += 1
                elif r1['min_dyad'] > r2['dyad']:
                    r_loc = {}
                    for k in list(nuc_df): r_loc[k] = np.nan
                    r_loc['dyad' + next_idx] = r2['dyad']
                    r_loc['occupancy' + next_idx] = r2['min_occ_score']
                    r_loc['pdyad' + next_idx] = r2['dyad_score']
                    r_loc['shift_' + curr_idx + next_idx] = np.nan
                    r_loc['chr'] = c
                    df = df.append(r_loc, ignore_index=True)
                    i2 += 1
                else:
                    r_loc = r1.to_dict()
                    r_loc['dyad' + next_idx] = np.nan
                    r_loc['occupancy' + next_idx] = np.nan
                    r_loc['pdyad' + next_idx] = np.nan
                    r_loc['shift_' + curr_idx + next_idx] = np.nan
                    df = df.append(r_loc, ignore_index=True) 
                    i1 += 1
                
            elif r1['curr_dyad'] < r2['dyad']:
                if r2['dyad'] <= r1['curr_dyad'] + max_nuc_dist:
                    r_loc = r1.to_dict()
                    r_loc['dyad' + next_idx] = r2['dyad']
                    r_loc['occupancy' + next_idx] = r2['min_occ_score']
                    r_loc['pdyad' + next_idx] = r2['dyad_score']
                    r_loc['shift_' + curr_idx + next_idx] = r2['dyad'] - r1['curr_dyad']
                    df = df.append(r_loc, ignore_index=True) 
                    i1 += 1
                    i2 += 1
                else:
                    r_loc = r1.to_dict()
                    r_loc['dyad' + next_idx] = np.nan
                    r_loc['occupancy' + next_idx] = np.nan
                    r_loc['pdyad' + next_idx] = np.nan
                    r_loc['shift_' + curr_idx + next_idx] = np.nan
                    df = df.append(r_loc, ignore_index=True) 
                    i1 += 1
            else:
                if r1['curr_dyad'] <= r2['dyad'] + max_nuc_dist:
                    r_loc = r1.to_dict()
                    r_loc['dyad' + next_idx] = r2['dyad']
                    r_loc['occupancy' + next_idx] = r2['min_occ_score']
                    r_loc['pdyad' + next_idx] = r2['dyad_score']
                    r_loc['shift_' + curr_idx + next_idx] = r2['dyad'] - r1['curr_dyad']
                    df = df.append(r_loc, ignore_index=True) 
                    i1 += 1
                    i2 += 1
                else:
                    r_loc = {}
                    for k in list(nuc_df): r_loc[k] = np.nan
                    r_loc['dyad' + next_idx] = r2['dyad']
                    r_loc['occupancy' + next_idx] = r2['min_occ_score']
                    r_loc['pdyad' + next_idx] = r2['dyad_score'] 
                    r_loc['shift_' + curr_idx + next_idx] = np.nan
                    r_loc['chr'] = c
                    df = df.append(r_loc, ignore_index=True)
                    i2 += 1

        for i in range(i1, l1):
            r_loc = nuc_df_chr.iloc[i].to_dict()
            r_loc['dyad' + next_idx] = np.nan
            r_loc['occupancy' + next_idx] = np.nan
            r_loc['pdyad' + next_idx] = np.nan
            r_loc['shift_' + curr_idx + next_idx] = np.nan
            df = df.append(r_loc, ignore_index=True) 
        for i in range(i2, l2):
            r_loc = {}
            r2 = nucschr.iloc[i]
            for k in list(nuc_df): r_loc[k] = np.nan
            r_loc['dyad' + next_idx] = r2['dyad']
            r_loc['occupancy' + next_idx] = r2['min_occ_score']
            r_loc['pdyad' + next_idx] = r2['dyad_score']
            r_loc['shift_' + curr_idx + next_idx] = np.nan
            r_loc['chr'] = c
            df = df.append(r_loc, ignore_index=True)


    dyads = sorted(list(filter(lambda x: x[:-1] == 'dyad', list(df))))
    df = df.sort_values(by=['chr', 'mean_dyad'] + dyads)
    df = df.reset_index(drop=True)
    df = df.drop(columns=['min_dyad', 'max_dyad', 'mean_dyad', 'curr_dyad'])
    # names = ["nuc_" + str(i) for i in range(len(df))]
    # df['name'] = names
    # df = df[['name', 'chr', 'dyadA', 'dyadB', 'occupancyA', 'occupancyB', 'shift']]
    return df


def nuc_map_pair(dirname1, dirname2, outdir, outfilename = 'nuc_map.csv'):
    configFile1 = dirname1 + "/config.ini"
    configFile2 = dirname2 + "/config.ini"
    outdir = outdir if outdir[-1] == '/' else outdir + '/'
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok = True)
    outfile = outdir + outfilename
    '''
    if os.path.isfile(outfile):
        nuc_df = pandas.read_csv(outfile, sep = '\t')
        return nuc_df
    '''
    config1 = ConfigParser() # SafeConfigParser()
    config1.read(configFile1)
    config2 = ConfigParser() # SafeConfigParser()
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

    nuc_df = match_pair_nucs(dirname1, dirname2, hmmconfig1, hmmconfig2, chrSizes)
    nuc_df.to_csv(outfile, sep = '\t', index = False)
    return nuc_df

def nuc_map_concat(nuc_df, dirname, outdir, outfilename = 'nuc_map.csv'):
    configFile = dirname + "/config.ini"
    outdir = outdir if outdir[-1] == '/' else outdir + '/' 
    os.makedirs(outdir, exist_ok = True)
    outfile = outdir + outfilename
    '''
    if os.path.isfile(outfile):
        nuc_df = pandas.read_csv(outfile, sep = '\t')
        return nuc_df
    '''
    config = ConfigParser() # SafeConfigParser()
    config.read(configFile)

    # get size of each chromosome
    chrSizes = {}
    fastaFile = config.get("main", "nucFile")
    fastaSeq = list(SeqIO.parse(open(fastaFile), 'fasta'))
    for fs in fastaSeq: chrSizes[fs.name] = len(fs.seq)

    bamFile = config.get("main", "bamFile")
    hmmconfigfile = config.get("main", "trainDir") + "/HMMconfig.pkl"
    hmmconfig = pickle.load(open(hmmconfigfile, "rb"), encoding = "latin1")

    nuc_df = match_new_nucs(nuc_df, dirname, hmmconfig, chrSizes)
    nuc_df.to_csv(outfile, sep = '\t', index = False)
    return nuc_df

def get_nuc_categories(nuc_df, outdir):
    outfile = outdir + 'nuc_map_categories.csv'
    '''
    if os.path.isfile(outfile):
        nuc_df = pandas.read_csv(outfile, sep = '\t')
        return nuc_df
    '''
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

def nuc_map_multiple(dirnames, outdir):
    outdir = outdir if outdir[-1] == '/' else outdir + '/' 
    outfile = outdir + 'nuc_map.csv'
    
    if os.path.isfile(outfile):
        nuc_df = pandas.read_csv(outfile, sep = '\t')
        return nuc_df
    
    if len(dirnames) == 2:
        return nuc_map_pair(dirnames[0], dirnames[1], outdir)

    nuc_df = nuc_map_pair(dirnames[0], dirnames[1], outdir, outfilename = 'nuc_map_01.csv')
    nuc_df = nuc_df.drop(columns = ['name'])
    curr_char = 'B'
    dyads = ['dyadA', 'dyadB']
    for i in range(1, len(dirnames) - 1):
        next_char = chr(ord(curr_char) + 1)
        nuc_df = nuc_map_concat(nuc_df, dirnames[i+1], outdir, outfilename = 'nuc_map_' + str(i) + str(i+1) + '.csv')

        '''
        nuc_df_new = nuc_df_new.rename(columns={'shift': 'shift_' + curr_char + next_char, 'dyadA': 'dyad' + curr_char, 'dyadB': 'dyad' + next_char, 'occupancyA': 'occupancy' + curr_char, 'occupancyB': 'occupancy' + next_char})
        nuc_df_new = nuc_df_new.drop(columns = ['name'])
        nuc_df_na = nuc_df[(nuc_df['dyad' + curr_char].isnull()) & (nuc_df['occupancy' + curr_char].isnull())]
        nuc_df_notna = nuc_df[(~nuc_df['dyad' + curr_char].isnull()) & (~nuc_df['occupancy' + curr_char].isnull())]

        nuc_df_new_na = nuc_df_new[(nuc_df_new['dyad' + curr_char].isnull()) & (nuc_df_new['occupancy' + curr_char].isnull())]
        nuc_df_new_notna = nuc_df_new[(~nuc_df_new['dyad' + curr_char].isnull()) & (~nuc_df_new['occupancy' + curr_char].isnull())]

        nuc_df = pandas.merge(nuc_df_notna, nuc_df_new_notna, on = ['chr', 'dyad' + curr_char, 'occupancy' + curr_char], how = 'inner')
        nuc_df = nuc_df.append(nuc_df_na, ignore_index = True)
        nuc_df = nuc_df.append(nuc_df_new_na, ignore_index = True)
        '''
        curr_char = next_char
        dyads.append('dyad' + curr_char)
        nuc_df = nuc_df.sort_values(['chr'] + dyads)
        nuc_df = nuc_df.reset_index(drop=True)
        
    names = ["nuc_" + str(i) for i in range(len(nuc_df))]
    nuc_df['name'] = names
    nuc_df.to_csv(outfile, sep = '\t', index = False)
    return nuc_df
    
def get_shifts(nuc_df):
    n = len(list(filter(lambda x: x.startswith('dyad'), list(nuc_df)))) - 1
    curr_char = 'A'
    shifts = []
    for i in range(1, n+1):
        shift_val = np.empty(len(nuc_df))
        shift_val[:] = np.nan
        next_char = chr(ord(curr_char) + 1)
        cond = (~nuc_df['dyad' + next_char].isnull()) & (~nuc_df['dyad' + curr_char].isnull())
        shift_val[cond] = nuc_df['dyad' + next_char][cond] -  nuc_df['dyad' + curr_char][cond]
        nuc_df['shift' + curr_char + next_char] = shift_val
        curr_char = next_char
    return nuc_df

def make_beds(nuc_df, idx, dyads, filename):
    f = open(filename, 'w')
    idx = idx.astype(bool)
    for i in np.arange(idx.shape[0])[idx]:
        r = nuc_df.iloc[i]
        chrm = r['chr']
        start = np.min(r[dyads]) - 73
        end = np.max(r[dyads]) + 74
        f.write(chrm + '\t' + str(int(start)) + '\t' + str(int(end)) + '\n')
    f.close()

def plot_occ(nuc_df, idx, dyads, filename):
    idx = idx.astype(bool)
    occ = ['occupancy' + x[4:] for x in dyads]
    o = []
    for i in np.arange(idx.shape[0])[idx]:
        r = nuc_df.iloc[i]
        o.append(list(r[occ]))

    import matplotlib.pyplot as plt
    import seaborn
    o = np.array(o)
    labels = ['0 mins', '7.5 mins', '60 mins']
    for i in range(len(occ)):
        seaborn.distplot(o[:, i], label = labels[i], hist = False)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def get_sign(n):
    if n < 0: return -1
    return 1

def check_linear_shift(shifts, shift_threshold):
    sgn = 0
    for s in shifts:
        if abs(s) <= shift_threshold: continue
        if sgn == 0: sgn = get_sign(s)
        elif sgn != get_sign(s):
            return False
    return True

def cluster_nuc_shifts(nuc_df, shift_threshold = 20):
    nuc_df = get_shifts(nuc_df)
    n = len(list(filter(lambda x: x.startswith('dyad'), list(nuc_df)))) - 1
    curr_char = 'A'
    shifts = []
    dyads = ['dyadA']
    for i in range(1, n+1):
        next_char = chr(ord(curr_char) + 1)
        shifts.append('shift' + curr_char + next_char)
        curr_char = next_char
        dyads.append('dyad' + curr_char)

    shift_arr = nuc_df[shifts].values
    shift_type = ['' for i in range(shift_arr.shape[0])]

    for i in range(shift_arr.shape[0]):
        if np.all(np.abs(shift_arr[i]) <= shift_threshold):
            shift_type[i] = '1_no_shift'
        elif np.any(np.isnan(shift_arr[i])):
            shift_type[i] = '4_not_always_present'
        elif check_linear_shift(shift_arr[i], shift_threshold):
            shift_type[i] = '2_directional_shift'
        else:
            shift_type[i] = '3_nondirectional_shift'

    nuc_df['shift_type'] = shift_type
    return nuc_df
    # linear_shifts = np.zeros(shift_arr.shape[0])
    # nonlinear_shifts = np.zeros(shift_arr.shape[0])
    # dropouts = np.zeros(shift_arr.shape[0])
    # gains = np.zeros(shift_arr.shape[0])
    # no_shifts = np.zeros(shift_arr.shape[0])
    # for i in range(shift_arr.shape[0]):
    #     if np.all(np.abs(shift_arr[i]) <= 20):
    #         no_shifts[i] = 1
    #     elif np.all(np.abs(shift_arr[i]) > 20) and abs(shift_arr[i, 0]) < abs(shift_arr[i, 1]): linear_shifts[i] = 1
    #     elif np.all(np.abs(shift_arr[i]) > 20) and abs(shift_arr[i, 0]) > abs(shift_arr[i, 1]): nonlinear_shifts[i] = 1
    #     elif np.any(np.isnan(shift_arr[i])): dropouts[i] = 1
        
    # print(np.sum(no_shifts))
    # print(np.sum(linear_shifts))
    # print(np.sum(nonlinear_shifts))
    # print(np.sum(dropouts))

    # plot_occ(nuc_df, no_shifts, dyads, filename='/usr/xtmp/sneha/tmpDir/no_shifts.png')
    # plot_occ(nuc_df, linear_shifts, dyads, filename='/usr/xtmp/sneha/tmpDir/linear_shifts.png')
    # plot_occ(nuc_df, nonlinear_shifts, dyads, filename='/usr/xtmp/sneha/tmpDir/nonlinear_shifts.png')
    # exit(0)
    
    # make_beds(nuc_df, no_shifts, dyads, filename='/usr/xtmp/sneha/tmpDir/no_shifts.bed')
    # make_beds(nuc_df, linear_shifts, dyads, filename='/usr/xtmp/sneha/tmpDir/linear_shifts.bed')
    # make_beds(nuc_df, nonlinear_shifts, dyads, filename='/usr/xtmp/sneha/tmpDir/nonlinear_shifts.bed')
    # make_beds(nuc_df, dropouts, dyads, filename='/usr/xtmp/sneha/tmpDir/dropouts.bed')

def cluster_nucs_occ_pdyad(nuc_df, k):
    occs = sorted(list(filter(lambda x: x.startswith('pdyad'), list(nuc_df))))
    occ_arr = nuc_df[occs].values
    occ_arr[occ_arr > 1] = 1
    occ_arr[np.isnan(occ_arr)] = 0
    nuc_occ = pandas.DataFrame(occ_arr, columns = occs)

    #do the clustering
    k_means = KMeans(n_clusters=k, random_state=9)
    k_means.fit(occ_arr)
    predict = k_means.fit_predict(occ_arr) # k_means.predict(occ_arr)
    # reorder cluster names
    new_predict = []
    for p in predict:
        if p == 0: new_predict.append(2)
        elif p == 2: new_predict.append(3)
        elif p == 3: new_predict.append(0)
        else: new_predict.append(1)

    predict = np.array(new_predict)
    
    nuc_occ['cluster'] = predict + 1
    
    nuc_df['occ_cluster'] = predict + 1 # nuc_occ['cluster']
    return nuc_df

def cluster_gene_nucs(nuc_df, outdir, plus_minus_ann, k, shifts_wrt_first=True):

    if not os.path.isfile(outdir + 'nuc_df_gene_properties.csv'):
        # plus_minus_ann = pandas.read_csv(plus_minus_ann_file, sep=',')
        get_gene_features(nuc_df, outdir, plus_minus_ann)
    df = pandas.read_csv(outdir + 'nuc_df_gene_properties.csv', sep='\t', index_col='gene')

    # make shifts with respect to first time point
    if shifts_wrt_first:
        df_p1 = df[df.columns[df.columns.str.startswith('p1_shift_')]]
        # get shifts wrt 0 mins
        p_col = ''
        for c in df_p1.columns:
            if p_col != '':
                df[c] = df[c] + df[p_col]
            p_col = c
        
        
        df_m1 = df[df.columns[df.columns.str.startswith('m1_shift_')]]
        # get shifts wrt 0 mins
        p_col = ''
        for c in df_m1.columns:
            if p_col != '':
                df[c] = df[c] + df[p_col]
            p_col = c


    pdyad_cols = sorted(list(filter(lambda x: '_pdyad' in x, df.columns)))

    df_scaled = df.fillna(0)

    scaler = StandardScaler()
    df_scaled = pandas.DataFrame(scaler.fit_transform(df.fillna(0)), index=df.index, columns=df.columns)

    k_means = KMeans(n_clusters=k, random_state=9)
    df_scaled['cluster'] = k_means.fit_predict(df_scaled.fillna(0))
    df['cluster'] = df_scaled['cluster']

    return df

def get_gene_features(nuc_df, outdir, plus_minus_ann):
    genes = sorted(list(set(plus_minus_ann['ORF'])))
    occs = sorted(list(filter(lambda x: x.startswith('occupancy'), list(nuc_df))))
    dyads = sorted(list(filter(lambda x: x.startswith('dyad'), list(nuc_df))))
    poccs = sorted(list(filter(lambda x: x.startswith('pdyad'), list(nuc_df))))
    shift_all_p1 = []
    shift_all_m1 = []
    poccs_all_p1 = []
    poccs_all_m1 = []
    poccs_orf = []
    poccs_promoter = []
    nuc_density_orf = []
    nuc_density_promoter = []
    
    occs_p1 = ['p1_' + o for o in occs]
    occs_m1 = ['m1_' + o for o in occs]
    occs_orf = ['orf_' + o for o in occs]
    occs_promoter = ['orf_' + o for o in occs]
    
    poccs_p1 = ['p1_' + o for o in poccs]
    poccs_m1 = ['m1_' + o for o in poccs]
    poccs_orf = ['orf_' + o for o in poccs]
    poccs_promoter = ['promoter_' + o for o in poccs]
    
    
    n = len(dyads)
    for i in range(1, n):
        shift_all_p1.append('p1_shift_' + chr(ord('A') + i - 1) + chr(ord('A') + i))
        shift_all_m1.append('m1_shift_' + chr(ord('A') + i - 1) + chr(ord('A') + i))
    for i in range(n):
        nuc_density_orf.append('nuc_density_orf_' + chr(ord('A') + i))
        nuc_density_promoter.append('nuc_density_promoter_' + chr(ord('A') + i))
    ispresent_p1 = ['p1_isPresent_' + chr(ord('A') + i) for i in range(n)]
    ispresent_m1 = ['m1_isPresent_' + chr(ord('A') + i) for i in range(n)]
    ispresent_orf = ['orf_isPresent_' + chr(ord('A') + i) for i in range(n)]
    ispresent_promoter = ['promoter_isPresent_' + chr(ord('A') + i) for i in range(n)]
    # nuctypepresent = ['nuc_type_+1', 'nuc_type_-1']
    flag = 0
    df_outfile = outdir + 'nuc_df_gene_properties.csv'
    for gene in genes:
        print(gene)
        strand = 1 if gene.split('-')[0][-1] == 'W' else -1
        cond_p1 = [False if type(r['+1_nuc']) != type('A') else (gene in r['+1_nuc'].split(',')) for i, r in nuc_df.iterrows()]
        cond_m1 = [False if type(r['-1_nuc']) != type('A') else (gene in r['-1_nuc'].split(',')) for i, r in nuc_df.iterrows()]
        cond_orf_transcript = [False if type(r['ORF_transcript_nuc']) != type('A') else (gene in r['ORF_transcript_nuc'].split(',')) for i, r in nuc_df.iterrows()]
        d = {}
        if np.sum(cond_p1) == 0 and np.sum(cond_m1) == 0 and np.sum(cond_orf_transcript) == 0: continue

        # +1 nuc
        if np.sum(cond_p1) == 0:
            for sa in shift_all_p1:
                d[sa] = np.nan
            for p in poccs_p1:
                d[p] = np.nan
        else:
            nd = nuc_df[cond_p1].iloc[0]
            for sa in shift_all_p1:
                d[sa] = nd[sa[3:]] * strand
            for p in poccs_p1:
                d[p] = nd[p.split('_')[1]]
        # -1 nuc
        if np.sum(cond_m1) == 0:
            for sa in shift_all_m1:
                d[sa] = np.nan
            for p in poccs_m1:
                d[p] = np.nan
        else:
            nd = nuc_df[cond_m1].iloc[0]
            for sa in shift_all_m1:
                d[sa] = nd[sa[3:]] * strand
            for p in poccs_m1:
                d[p] = nd[p.split('_')[1]]
        # ORF nuc
        if np.sum(cond_orf_transcript) == 0:
            for p in poccs_orf:
                d[p] = np.nan
        else:
            nd = nuc_df[cond_orf_transcript].fillna(0)
            for p in poccs_orf:
                d[p] = np.mean(nd[p.split('_')[1]])

        d['gene'] = gene
        df = pandas.DataFrame(columns=['gene'] + shift_all_p1 + shift_all_m1 + poccs_p1 + poccs_m1 + poccs_orf)
        df = df.append(d, ignore_index=True)
        
        if flag == 0:
            flag = 1
            df.to_csv(df_outfile, sep='\t', index=False)
        else:
            df.to_csv(df_outfile, sep='\t', index=False, mode='a', header=False)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python nuc_map_same_analysis.py <dirname1> <dirname2> <outdir>")
        exit(0)
    dirname1 = (sys.argv)[1]
    dirname2 = (sys.argv)[2]
    outdir = (sys.argv)[3]

    nuc_df = nuc_map(dirname1, dirname2, outdir)
    nuc_df = get_nuc_categories(nuc_df, outdir)
