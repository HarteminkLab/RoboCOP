import numpy as np
import pandas
import seaborn
import sys
from Bio import SeqIO
from configparser import ConfigParser
# from configparser import SafeConfigParser
import pickle
import math
import pysam
import os
import re
import h5py
import glob
import scipy

# get posterior table for a nucleosome locus
def get_posterior_nuc(allinfofiles, dshared, nuc, coords):
    chrm = nuc['chr']
    start = max(0, nuc['dyad'] - 73)
    end = nuc['dyad'] + 73

    idxs = list(coords[(coords['chr'] == chrm) & (start <= coords['end']) & (end >= coords['start'])].index)
    ptable = np.zeros((end - start + 1, dshared['n_states']))
    counts = np.zeros(end - start + 1)
    
    for infofile in allinfofiles:
        for i in range(len(idxs)):
            idx = idxs[i]
            f = h5py.File(infofile, mode = 'r')
            k = 'segment_' + str(idx)
            if k not in f.keys():
                f.close()
                continue
            dp = f[k + '/posterior'][:]
            f.close()
            dp_start = max(0, start - coords.loc[idx]['start'])
            dp_end = min(end - coords.loc[idx]['start'] + 1, coords.loc[idx]['end'] - coords.loc[idx]['start'] + 1)
            ptable_start = max(0, coords.loc[idx]['start'] - start)
            ptable_end = ptable_start + dp_end - dp_start
            print("dp start end:", dp_start, dp_end)
            print("ptable start end:", ptable_start, ptable_end)
            ptable[ptable_start : ptable_end] += dp[dp_start : dp_end, :] 
            counts[ptable_start : ptable_end] += 1
    ptable[counts > 0, :] = ptable[counts > 0, :] / counts[counts > 0][:, np.newaxis]
    return ptable

# get absolute max diff score within each nucleosome
# present in both data sets
def get_tf_diff_dist_in_not_null(dirname1, dirname2, outdir, nuc_df, hmmconfig):
    if os.path.isfile(outdir + 'tf_posterior_in_nuc_not_null.csv'):
        posterior_tf_diff = pandas.read_csv(outdir + 'tf_posterior_in_nuc_not_null.csv', sep = '\t')
        return posterior_tf_diff
    allinfofiles1 = glob.glob(dirname1 + 'tmpDir/info*.h5')
    allinfofiles2 = glob.glob(dirname2 + 'tmpDir/info*.h5')
    coords = pandas.read_csv(dirname1 + "coords.tsv", sep = "\t")
    nuc_df_not_null = nuc_df[nuc_df['category'] == 'not_null']
    nuc_df_not_null = nuc_df_not_null.reset_index()
    # nuc_df_not_null = nuc_df_not_null[:1000]
    
    posterior_tf_diff = np.zeros((len(nuc_df_not_null), hmmconfig['n_tfs']))

    flag = np.zeros(len(nuc_df_not_null))
    for infofile1 in allinfofiles1:
        f1 = h5py.File(infofile1, mode = 'r')
        for infofile2 in allinfofiles2:
            f2 = h5py.File(infofile2, mode = 'r')
            segments = list(set(f1.keys()).intersection(set(f2.keys())))
            for segment in segments:
                chrm = f1[segment].attrs['chr']
                start = f1[segment].attrs['start']
                end = f1[segment].attrs['end']
                nuc_segment = nuc_df_not_null[(nuc_df_not_null['chr'] == chrm) & (nuc_df_not_null['dyad'] - 73 >= start) & (nuc_df_not_null['dyad'] + 73 <= end)]
                if nuc_segment.empty: continue
                flag[nuc_segment.index] = 1

                ptable1 = f1[segment + '/posterior'][:]
                ptable2 = f2[segment + '/posterior'][:]
                ptable_diff = ptable2 - ptable1
                for i, r in nuc_segment.iterrows():
                    print(i)
                    p_start = min(r['dyadB'] - 73, r['dyadA']) if r['dyadA'] < r['dyadB'] else min(r['dyadA'] - 73, r['dyadB']) # r['dyad'] - 73 - start
                    p_end = max(r['dyadA'] + 73, r['dyadB']) if r['dyadA'] < r['dyadB'] else max(r['dyadB'] + 73, r['dyadA']) # r['dyad'] + 73 - start + 1
                    p_start = int(p_start - start)
                    p_end = int(p_end - start + 1)
                    p_diff_nuc = ptable_diff[p_start : p_end, :]
                    for tf_idx in range(hmmconfig['n_tfs']):
                        diff_score = np.sum(p_diff_nuc[:, [hmmconfig['tf_starts'][tf_idx], hmmconfig['tf_lens'][tf_idx]]], axis = 1)
                        if posterior_tf_diff[i, tf_idx] != 0:
                            posterior_tf_diff[i, tf_idx] = 0.5*(diff_score[np.argmax(np.abs(diff_score))] + posterior_tf_diff[i, tf_idx])
                        else:
                            posterior_tf_diff[i, tf_idx] = diff_score[np.argmax(np.abs(diff_score))]
            f2.close()
        f1.close()

    posterior_tf_diff = pandas.DataFrame(posterior_tf_diff, columns = list(hmmconfig['tfs']), index = nuc_df_not_null['name'])
    posterior_tf_diff.to_csv(outdir + 'tf_posterior_in_nuc_not_null.csv', sep = '\t', index = False)
    return posterior_tf_diff

def get_segment_file_handle(dirname, segment):
    allinfofiles = glob.glob(dirname + 'tmpDir/info*.h5')
    for infofile in allinfofiles:
        f = h5py.File(infofile, mode = 'r')
        if segment in f.keys(): return f
        f.close()

def calc_pvals_segment(p_diff, posterior_tf_diff, hmmconfig, chrm, start, end, prob_A, prob_B, p_val_cutoff = 0.05):
    p_scores = np.zeros((p_diff.shape[0], hmmconfig['n_tfs']))
    for tf_idx in range(hmmconfig['n_tfs']):
        p_scores[:, tf_idx] = np.sum(p_diff[:, [hmmconfig['tf_starts'][tf_idx], hmmconfig['tf_starts'][tf_idx] + hmmconfig['tf_lens'][tf_idx]]], axis = 1)

    z_scores = (p_scores.T - np.mean(posterior_tf_diff, axis = 0)[:, np.newaxis]) / np.std(posterior_tf_diff, axis = 0)[:, np.newaxis]
    z_scores = z_scores.T

    # two-sided p-values
    p_vals = scipy.stats.norm.sf(abs(z_scores))*2

    df = pandas.DataFrame(columns = ['chr', 'start', 'end', 'TF', 'p-value', 'Z-score', 'prob_diff', 'prob_A', 'prob_B'])

    for tf_idx in range(hmmconfig['n_tfs']):
        for i in range(p_vals.shape[0]):
            if p_vals[i, tf_idx] < p_val_cutoff and abs(p_diff[i, tf_idx]) > 1e-3 and prob_A[i, tf_idx] <= 1 and prob_B[i, tf_idx] <= 1: # p_vals[i, tf_idx] > 0 and p_vals[i, tf_idx] <= p_val_cutoff and 
                # print(chrm, start, end, p_vals.shape, z_scores.shape, p_diff.shape, prob_A.shape, prob_B.shape)
                df = df.append({'chr': chrm, 'start': i + start + 1, 'end': i + start + hmmconfig['tf_lens'][tf_idx] + 1, 'TF': hmmconfig['tfs'][tf_idx], 'p-value': p_vals[i, tf_idx], 'Z-score': z_scores[i, tf_idx], 'prob_diff': p_diff[i, tf_idx], 'prob_A': prob_A[i, tf_idx], 'prob_B': prob_B[i, tf_idx]}, ignore_index = True)

    return df

def calculate_pvals(dirname1, dirname2, coords, outdir, posterior_tf_diff, hmmconfig):
    output_path = outdir + 'tf_diff_pvals.csv'
    if os.path.isfile(output_path):
        df = pandas.read_csv(output_path, sep = '\t')
        return df
    
    chrs = list(set(coords['chr']))
    write_mode = 'w'
    for chrm in chrs:
        coords_chr = coords[coords['chr'] == chrm]
        coords_chr = coords_chr.sort_values(by = ['start'])
        pend = -1
        for i, r in coords_chr.iterrows():
            segment = 'segment_' + str(i)
            f1 = get_segment_file_handle(dirname1, segment)
            f2 = get_segment_file_handle(dirname2, segment)

            if pend == -1:
                p_diff = f2[segment + '/posterior'][:] - f1[segment + '/posterior'][:]
                prob_A = f1[segment + '/posterior'][:]
                prob_B = f2[segment + '/posterior'][:]
                pend = r['end']
                pstart = r['start']
            elif pend >= r['start']:
                pd = f2[segment + '/posterior'][:] - f1[segment + '/posterior'][:]
                p_A = f1[segment + '/posterior'][:]
                p_B = f2[segment + '/posterior'][:]
                overlap = pend - r['start'] + 1
                p_diff = np.concatenate([p_diff[:-overlap, :], (p_diff[-overlap:, :] + pd[:overlap, :])/2, pd[overlap:, :]], axis = 0)
                prob_A = np.concatenate([prob_A[:-overlap, :], (prob_A[-overlap:, :] + p_A[:overlap, :])/2, p_A[overlap:, :]], axis = 0)
                prob_B = np.concatenate([prob_B[:-overlap, :], (prob_B[-overlap:, :] + p_B[:overlap, :])/2, p_B[overlap:, :]], axis = 0)
                pend = r['end']
            else:
                df = calc_pvals_segment(p_diff, posterior_tf_diff, hmmconfig, chrm, pstart, pend, prob_A, prob_B)
                if not df.empty:
                    df.to_csv(output_path, mode=write_mode, header=write_mode == 'w', index = False, sep = '\t')
                    write_mode = 'a'
                p_diff = f2[segment + '/posterior'][:] - f1[segment + '/posterior'][:]
                prob_A = f1[segment + '/posterior'][:]
                prob_B = f2[segment + '/posterior'][:]
                pend = r['end']
                pstart = r['start']

        if pend != -1:
            df = calc_pvals_segment(p_diff, posterior_tf_diff, hmmconfig, chrm, pstart, pend, prob_A, prob_B)
            if not df.empty:
                df.to_csv(output_path, mode=write_mode, header=write_mode == 'w', index = False, sep = '\t')
                write_mode = 'a'
    return df

def get_tf_diff_in_diff_nucs(outdir):
    nuc_df = pandas.read_csv(outdir + 'nuc_map_categories.csv', sep = '\t')
    tf_diff = pandas.read_csv(outdir + 'tf_diff_pvals.csv', sep = '\t')

    inside_nuc = ['' for i in range(len(tf_diff))]
    for i, r in tf_diff.iterrows():
        nuc = nuc_df[(nuc_df['chr'] == tf_diff['chr']) & (nuc_df['dyad'] - 73 <= tf_diff['end']) & (nuc_df['dyad'] + 74 >= tf_diff['start'])]
        if nuc.empty: continue
        inside_nuc[i] = nuc.iloc[0]['category']

    tf_diff['within_nucleosome'] = inside_nuc
    print(tf_diff)
    
def get_tf_diff_map(dirname1, dirname2, outdir):
    dirname1 = dirname1 if dirname1[-1] == '/' else dirname1 + '/'
    dirname2 = dirname2 if dirname2[-1] == '/' else dirname2 + '/'
    outdir = outdir if outdir[-1] == '/' else outdir + '/'
        
    nuc_df = pandas.read_csv(outdir + 'nuc_map_categories.csv', sep = '\t')
    coords = pandas.read_csv(dirname1 + 'coords.tsv', sep = '\t')
    
    configFile1 = dirname1 + "/config.ini"
    configFile2 = dirname2 + "/config.ini"
    config1 = ConfigParser() # SafeConfigParser()
    config1.read(configFile1)
    config2 = ConfigParser() # SafeConfigParser()
    config2.read(configFile2)

    hmmconfigfile1 = config1.get("main", "trainDir") + "/HMMconfig.pkl"
    hmmconfig1 = pickle.load(open(hmmconfigfile1, "rb"), encoding = "latin1")

    hmmconfigfile2 = config2.get("main", "trainDir") + "/HMMconfig.pkl"
    hmmconfig2 = pickle.load(open(hmmconfigfile2, "rb"), encoding = "latin1")

    posterior_tf_diff = get_tf_diff_dist_in_not_null(dirname1, dirname2, outdir, nuc_df, hmmconfig1)
    print(posterior_tf_diff)
    df = calculate_pvals(dirname1, dirname2, coords, outdir, posterior_tf_diff, hmmconfig1)
    print(df)
    # get_tf_diff_in_diff_nucs(outdir)
    
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python nuc_map_same_analysis.py <dirname1> <dirname2> <outdir>")

    dirname1 = (sys.argv)[1]
    dirname2 = (sys.argv)[2]
    outdir = (sys.argv)[3]
    get_tf_diff_map(dirname1, dirname2, outdir)
    
