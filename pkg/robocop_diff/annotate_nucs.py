import os
import math
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn

def annotate_plus1_minus1_nucs(nuc_df, plus_minus_ann, outdir):
    if os.path.isfile(outdir + 'nuc_diff_with_pplus_minus_nucs_dyn.csv'):
        nuc_df = pandas.read_csv(outdir + 'nuc_diff_with_pplus_minus_nucs_dyn.csv', sep='\t') 
        return nuc_df
    
    plus1 = [None for i in range(len(nuc_df))]
    minus1 = [None for i in range(len(nuc_df))]
    dyads = sorted(list(filter(lambda x: x.startswith('dyad'), list(nuc_df))))
    # plus_minus_ann = pandas.read_csv('/usr/xtmp/sneha/Chereji_2018/13059_2018_1398_MOESM2_ESM.csv', sep=',')
    p_chr = ''
    for i, r in nuc_df.iterrows():
        if p_chr != r['chr']:
            plus_minus_ann_chr = plus_minus_ann[plus_minus_ann['Chr'] == r['chr']]
            p_chr = r['chr']

        for dyad in dyads:
            if r[dyad] is None: continue 
            pm_plus = plus_minus_ann_chr[(plus_minus_ann_chr['+1 nucleosome'] >= r[dyad] - 50) & (plus_minus_ann_chr['+1 nucleosome'] <= r[dyad] + 50)]
            if not pm_plus.empty:
                plus1[i] = pm_plus.iloc[0]['ORF']
                break

        for dyad in dyads:
            if r[dyad] is None: continue
            pm_minus = plus_minus_ann_chr[(plus_minus_ann_chr['-1 nucleosome'] >= r[dyad] - 50) & (plus_minus_ann_chr['-1 nucleosome'] <= r[dyad] + 50)]
            if not pm_minus.empty:
                minus1[i] = pm_minus.iloc[0]['ORF']
                break
    nuc_df['+1_nuc'] = plus1
    nuc_df['-1_nuc'] = minus1
    # nuc_df = nuc_df.astype({'+1_nuc': 'float', '-1_nuc': 'float'}) 
    nuc_df.to_csv(outdir + 'nuc_diff_with_pplus_minus_nucs_dyn.csv', sep='\t', index=False)
    return nuc_df

def get_promoter(pm, outdir):
    if os.path.isfile(outdir + 'nucs_with_promoter.csv'):
        pm = pandas.read_csv(outdir + 'nucs_with_promoter.csv', sep='\t')
        return pm
    
    p_start = np.zeros(len(pm))
    p_end = np.zeros(len(pm))

    for i, r in pm.iterrows():
        ps = max(1, r['TSS'] - 2000) if r['Strand'] == 1 else r['TSS'] + 1
        pe = r['TSS'] - 1 if r['Strand'] == 1 else r['TSS'] + 2000

        if r['Strand'] == 1:
            # pmf_p = pm[(pm['Chr'] == r['Chr']) & (pm['TSS'] < pe) & (pm['TTS'] < pe) & (pm['ORF Start'] < pe) & (pm['ORF End'] < pe) & (pm['+1 nucleosome'] - 73 < pe) & (pm['-1 nucleosome'] - 73 < pe)] 
            pmf_p = pm[(pm['Chr'] == r['Chr']) & (pm['TSS'] < pe) & (pm['TTS'] < pe) & (pm['ORF Start'] < pe) & (pm['ORF End'] < pe) & (pm['ORF'] != r['ORF'])] 
            if not pmf_p.empty:
                pmf_p = np.max(pmf_p[['TSS', 'TTS', 'ORF Start', 'ORF End']].values)
                ps = max(ps, pmf_p)

        else:
            # pmf_n = pm[(pm['Chr'] == r['Chr']) & (pm['TSS'] > ps) & (pm['TTS'] > ps) & (pm['ORF Start'] > ps) & (pm['ORF End'] > ps) & (pm['+1 nucleosome'] + 73 > ps) & (pm['-1 nucleosome'] + 73 > ps)]
            pmf_n = pm[(pm['Chr'] == r['Chr']) & (pm['TSS'] > ps) & (pm['TTS'] > ps) & (pm['ORF Start'] > ps) & (pm['ORF End'] > ps) & (pm['ORF'] != r['ORF'])]
            if not pmf_n.empty:
                pmf_n = np.min(pmf_n[['TSS', 'TTS', 'ORF Start', 'ORF End']].values)
                pe = min(pe, pmf_n)

        p_start[i] = ps
        p_end[i] = pe

    pm['Promoter2k_start'] = p_start
    pm['Promoter2k_end'] = p_end
    pm.to_csv(outdir + 'nucs_with_promoter.csv', sep='\t', index=False)
    return pm

def get_downstream(pm, outdir):
    if os.path.isfile(outdir + 'nucs_with_downstream.csv'):
        pm = pandas.read_csv(outdir + 'nucs_with_downstream.csv', sep='\t')
        return pm
    
    p_start = np.zeros(len(pm))
    p_end = np.zeros(len(pm))

    for i, r in pm.iterrows():
        ps = r['TTS'] + 1 if r['Strand'] == 1 else max(1, r['TTS'] - 500)
        pe = r['TTS'] + 500 if r['Strand'] == 1 else r['TTS'] - 1

        if r['Strand'] == 1:
            # pmf_p = pm[(pm['Chr'] == r['Chr']) & (pm['TSS'] < pe) & (pm['TTS'] < pe) & (pm['ORF Start'] < pe) & (pm['ORF End'] < pe) & (pm['+1 nucleosome'] - 73 < pe) & (pm['-1 nucleosome'] - 73 < pe)] 
            pmf_p = pm[(pm['Chr'] == r['Chr']) & (pm['TSS'] < pe) & (pm['TTS'] < pe) & (pm['ORF Start'] < pe) & (pm['ORF End'] < pe) & (pm['ORF'] != r['ORF'])] 
            if not pmf_p.empty:
                pmf_p = np.max(pmf_p[['TSS', 'TTS', 'ORF Start', 'ORF End']].values)
                ps = max(ps, pmf_p)

        else:
            # pmf_n = pm[(pm['Chr'] == r['Chr']) & (pm['TSS'] > ps) & (pm['TTS'] > ps) & (pm['ORF Start'] > ps) & (pm['ORF End'] > ps) & (pm['+1 nucleosome'] + 73 > ps) & (pm['-1 nucleosome'] + 73 > ps)]
            pmf_n = pm[(pm['Chr'] == r['Chr']) & (pm['TSS'] > ps) & (pm['TTS'] > ps) & (pm['ORF Start'] > ps) & (pm['ORF End'] > ps) & (pm['ORF'] != r['ORF'])]
            if not pmf_n.empty:
                pmf_n = np.min(pmf_n[['TSS', 'TTS', 'ORF Start', 'ORF End']].values)
                pe = min(pe, pmf_n)

        p_start[i] = ps
        p_end[i] = pe

    pm['Downstream500_start'] = p_start
    pm['Downstream500_end'] = p_end
    pm.to_csv(outdir + 'nucs_with_downstream.csv', sep='\t', index=False)
    return pm

def annotate_promoter_nucs(nuc_df, plus_minus_ann, outdir):
    if os.path.isfile(outdir + 'nuc_diff_with_pplus_minus_promoter_nucs_dyn.csv'):
        nuc_df = pandas.read_csv(outdir + 'nuc_diff_with_pplus_minus_promoter_nucs_dyn.csv', sep='\t') 
        return nuc_df

    ndrs = [None for i in range(len(nuc_df))]
    dyads = sorted(list(filter(lambda x: x.startswith('dyad'), list(nuc_df))))
    # plus_minus_ann = pandas.read_csv('/usr/xtmp/sneha/Chereji_2018/13059_2018_1398_MOESM2_ESM.csv', sep=',')
    plus_minus_ann = get_promoter(plus_minus_ann, outdir)
    # plus_minus_ann['NDR_start'] = plus_minus_ann['NDR Center'] - 0.5*plus_minus_ann['NDR Width']
    # plus_minus_ann['NDR_end'] = plus_minus_ann['NDR Center'] + 0.5*plus_minus_ann['NDR Width']
    p_chr = ''
    for i, r in nuc_df.iterrows():
        if type(r['+1_nuc']) == type("A") or type(r['-1_nuc']) == type("A"):
            continue
        if p_chr != r['chr']:
            plus_minus_ann_chr = plus_minus_ann[plus_minus_ann['Chr'] == r['chr']]
            p_chr = r['chr']
        ndr = []
        for dyad in dyads:
            if r[dyad] is None: continue 
            pm = plus_minus_ann_chr[(plus_minus_ann_chr['Promoter2k_start'] < r[dyad]) & (plus_minus_ann_chr['Promoter2k_end'] > r[dyad])]
            if pm.empty: continue
            for j, s in pm.iterrows():
                ndr.append(s['ORF'])
        if ndr == []: continue
        ndr = sorted(list(set(ndr)))
        ndr = ','.join(ndr)
        ndrs[i] = ndr

    nuc_df['Promoter_nuc'] = ndrs
    nuc_df.to_csv(outdir + 'nuc_diff_with_pplus_minus_promoter_nucs_dyn.csv', sep='\t', index=False)
    return nuc_df


def annotate_downstream_nucs(nuc_df, plus_minus_ann, outdir):
    if os.path.isfile(outdir + 'nuc_diff_with_pplus_minus_promoter_downstream_nucs_dyn.csv'):
        nuc_df = pandas.read_csv(outdir + 'nuc_diff_with_pplus_minus_promoter_downstream_nucs_dyn.csv', sep='\t') 
        return nuc_df

    ndrs = [None for i in range(len(nuc_df))]
    dyads = sorted(list(filter(lambda x: x.startswith('dyad'), list(nuc_df))))
    # plus_minus_ann = pandas.read_csv('/usr/xtmp/sneha/Chereji_2018/13059_2018_1398_MOESM2_ESM.csv', sep=',')
    plus_minus_ann = get_downstream(plus_minus_ann, outdir)
    p_chr = ''
    for i, r in nuc_df.iterrows():
        if type(r['+1_nuc']) == type("A") or type(r['-1_nuc']) == type("A") or type(r['Promoter_nuc']) == type("A") or type(r['ORF_transcript_nuc']) == type("A"):
            continue
        if p_chr != r['chr']:
            plus_minus_ann_chr = plus_minus_ann[plus_minus_ann['Chr'] == r['chr']]
            p_chr = r['chr']
        ndr = []
        for dyad in dyads:
            if r[dyad] is None: continue 
            pm = plus_minus_ann_chr[(plus_minus_ann_chr['Downstream500_start'] < r[dyad]) & (plus_minus_ann_chr['Downstream500_end'] > r[dyad])]
            if pm.empty: continue
            for j, s in pm.iterrows():
                ndr.append(s['ORF'])
        if ndr == []: continue
        ndr = sorted(list(set(ndr)))
        ndr = ','.join(ndr)
        ndrs[i] = ndr

    nuc_df['Downstream_nuc'] = ndrs
    nuc_df.to_csv(outdir + 'nuc_diff_with_pplus_minus_promoter_downstream_nucs_dyn.csv', sep='\t', index=False)
    return nuc_df

def annotate_transcript_ORF_nucs(nuc_df, plus_minus_ann, outdir):
    if os.path.isfile(outdir + 'nuc_diff_with_pplus_minus_ndr_orf_nucs_dyn.csv'):
        nuc_df = pandas.read_csv(outdir + 'nuc_diff_with_pplus_minus_ndr_orf_nucs_dyn.csv', sep='\t') 
        return nuc_df
    orfs = [None for i in range(len(nuc_df))]
    dyads = sorted(list(filter(lambda x: x.startswith('dyad'), list(nuc_df))))
    # plus_minus_ann = pandas.read_csv('/usr/xtmp/sneha/Chereji_2018/13059_2018_1398_MOESM2_ESM.csv', sep=',')
    plus_minus_ann['ot_start'] = np.min(plus_minus_ann[['ORF Start', 'ORF End', 'TSS', 'TTS']], axis=1)
    plus_minus_ann['ot_end'] = np.max(plus_minus_ann[['ORF Start', 'ORF End', 'TSS', 'TTS']], axis=1)

    p_chr = ''
    for i, r in nuc_df.iterrows():
        if type(r['+1_nuc']) == type("A") or type(r['-1_nuc']) == type("A") or type(r['Promoter_nuc']) == type("A"):
            continue
        if p_chr != r['chr']:
            plus_minus_ann_chr = plus_minus_ann[plus_minus_ann['Chr'] == r['chr']]
            p_chr = r['chr']

        orf = []
        for dyad in dyads:
            if r[dyad] is None: continue 
            pm = plus_minus_ann_chr[(plus_minus_ann_chr['ot_start'] < r[dyad]) & (plus_minus_ann_chr['ot_end'] > r[dyad])]
            if pm.empty: continue
            for j, s in pm.iterrows():
                orf.append(s['ORF'])
        if orf == []: continue
        orf = sorted(list(set(orf)))
        orf = ','.join(orf)
        orfs[i] = orf

    nuc_df['ORF_transcript_nuc'] = orfs
    nuc_df.to_csv(outdir + 'nuc_diff_with_pplus_minus_ndr_orf_nucs_dyn.csv', sep='\t', index=False)
    return nuc_df
    
def ann_nucs(nuc_df, plus_minus_file, outdir):
    plus_minus_ann = pandas.read_csv(plus_minus_file, sep=',')
    nuc_df = annotate_plus1_minus1_nucs(nuc_df, plus_minus_ann, outdir)
    nuc_df = annotate_promoter_nucs(nuc_df, plus_minus_ann, outdir)
    nuc_df = annotate_transcript_ORF_nucs(nuc_df, plus_minus_ann, outdir)
    nuc_df = annotate_downstream_nucs(nuc_df, plus_minus_ann, outdir)
    return nuc_df


def plot_nuc_anns(nuc_df, filename):

    shift_types = pandas.Series(['no_shift' for i in range(len(nuc_df))])
    shift_types[nuc_df['shift_type'] == 'linear_shift'] = 'directional_shift'
    shift_types[nuc_df['shift_type'] == 'nonlinear_shift'] = 'nondirectional_shift'
    shift_types[nuc_df['shift_type'] == 'depleted'] = 'not_always_present'
    nuc_df['shift_type'] = shift_types

    location = pandas.Series(['Intergenic_nuc' for i in range(len(nuc_df))])
    # location[(nuc_df['+1_nuc'].notna()) & (nuc_df['+1_nuc'].isna())] = '+1_nuc_only'
    # location[(nuc_df['+1_nuc'].isna()) & (nuc_df['-1_nuc'].notna())] = '-1_nuc_only'
    # location[(nuc_df['-1_nuc'].notna()) & (nuc_df['-1_nuc'].notna())] = '+1_-1_nuc_both'

    location[np.array([type(r['+1_nuc']) == type('A') and type(r['-1_nuc']) == type('A') for i, r in nuc_df.iterrows()]).astype(bool)] = '+1_-1_nuc_both'
    location[np.array([type(r['+1_nuc']) == type('A') and type(r['-1_nuc']) != type('A') for i, r in nuc_df.iterrows()]).astype(bool)] = '+1_nuc_only'
    location[np.array([type(r['+1_nuc']) != type('A') and type(r['-1_nuc']) == type('A') for i, r in nuc_df.iterrows()]).astype(bool)] = '-1_nuc_only'

    location[nuc_df['Promoter_nuc'].notna()] = 'Promoter_nuc'
    location[nuc_df['ORF_transcript_nuc'].notna()] = 'ORF_transcript_nuc'
    location[nuc_df['Downstream_nuc'].notna()] = 'Downstream_nuc'
    nuc_df['location'] = location
    print("+1 only:", len(nuc_df[nuc_df['location'] == '+1_nuc_only']))
    print("-1 only:", len(nuc_df[nuc_df['location'] == '-1_nuc_only']))
    print("+1 -1 both:", len(nuc_df[nuc_df['location'] == '+1_-1_nuc_both']))
    '''
    seaborn.catplot(x='occ_cluster', hue='shift_type', col='location', data=nuc_df, kind='count', sharey=False, log=True)
    plt.savefig('/usr/xtmp/sneha/tmpDir/tmp.png')
    plt.close()
    '''
    occs = sorted(list(set(nuc_df['occ_cluster'])), reverse=True)
    shifts = sorted(list(set(nuc_df['shift_type'])))

    pairs = []
    colors = plt.cm.Set2.colors
    nuc_pos = ['+1_nuc_only', '-1_nuc_only', '+1_-1_nuc_both', 'Promoter_nuc', 'ORF_transcript_nuc', 'Downstream_nuc', 'Intergenic_nuc']

    # plt.figure(figsize=(4, 6))
    fig, ax = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
    for npos in range(len(nuc_pos)):
        nps = nuc_pos[npos]
        y = []
        y_p = []
        x = []
        sizes = []
        for o in occs:
            for s in shifts:
                nd = nuc_df[(nuc_df['occ_cluster'] == o) & (nuc_df['shift_type'] == s)]
                if nd.empty: continue
                x.append('C' + str(int(o)) + '_' + s + ' (' + str(len(nd)) + ' nucs)')
                y.append(len(nd[nd['location'] == nps]))
                y_p.append(len(nd[nd['location'] == nps]) / len(nd) * 100)
                sizes.append(len(nd))
        idx = np.argsort(np.array(sizes))
        x = np.array(x)[idx]
        y = np.array(y)[idx]
        y_p = np.array(y_p)[idx]
        if npos == 0:
            ax[0].barh(x, y, color=colors[npos])
            ax[1].barh(x, y_p, color=colors[npos])
            y_prev = y
            y_p_prev = y_p
        else:
            print(y_p)
            print(y_p_prev)
            ax[0].barh(x, y, left=y_prev, color=colors[npos])
            ax[1].barh(x, y_p, left=y_p_prev, color=colors[npos])
            y_prev += y
            y_p_prev += y_p
    print(y_p_prev)
    ax[0].set_ylabel('P(nuc)_cluster x shift_type')
    ax[0].set_xlabel('Count')
    ax[1].set_xlabel('Percentage')
    ax[0].legend(nuc_pos)
    plt.tight_layout()
    plt.savefig(filename) # '/usr/xtmp/sneha/tmpDir/tmp.png')
    plt.close()
               

def get_whole_genome_counts():

    d = {'+1_nuc_only': 570685, '-1_nuc_only': 504163, '+1_-1_nuc_both': 281637, 'Promoter': 1711755, 'ORF_transcript': 8145716, 'Downstream': 232879, 'Intergenic': 624491}
    k = ['+1_nuc_only', '-1_nuc_only', '+1_-1_nuc_both', 'Promoter', 'ORF_transcript', 'Downstream', 'Intergenic']
    total = sum(list(d.values()))
    colors = plt.cm.Set2.colors
    plt.figure(figsize=(10, 3))
    for i in range(len(k)):
        if i == 0:
            plt.barh(['genome'], [d[k[i]] / total * 100], color=colors[i])
            y_p = d[k[i]] / total * 100
        else:
            plt.barh(['genome'], [d[k[i]] / total * 100], left=[y_p], color=colors[i])
            y_p += d[k[i]] / total * 100
        print(y_p)
        
    plt.xlabel("Percentage")
    # plt.legend(k)
    plt.tight_layout()
    plt.savefig('/usr/xtmp/sneha/tmpDir/tmp.png')
    plt.close()
    return
    
    d = {'+1_nuc_only': 0, '-1_nuc_only': 0, '+1_-1_nuc_both': 0, 'Promoter': 0, 'ORF_transcript': 0, 'Downstream': 0, 'Intergenic': 0}
    plus_minus_ann = pandas.read_csv('/usr/xtmp/sneha/Chereji_2018/13059_2018_1398_MOESM2_ESM.csv', sep=',')
    promoter = pandas.read_csv('/usr/xtmp/sneha/Chereji_2018/nucs_with_promoter.csv', sep='\t')
    downstream = pandas.read_csv('/usr/xtmp/sneha/Chereji_2018/nucs_with_downstream.csv', sep='\t')
    chrSizes = open("/usr/project/compbio/sneha/genome/chromSizes/sacCer3.chrom.sizes", 'r').readlines()
    chrSizes = map(lambda x: x.strip().split(), chrSizes)
    chrSizes = filter(lambda x: x[0] != 'chrM' and x[0] != '2micron', chrSizes)
    chrSizes = map(lambda x: (x[0], int(x[1])), chrSizes)
    chrSizes = dict(chrSizes)
    genes_plus1 = []
    genes_minus1 = []

    for chrm in chrSizes:
        plus_minus_ann_chr = plus_minus_ann[plus_minus_ann['Chr'] == chrm]
        promoter_chr = promoter[promoter['Chr'] == chrm]
        downstream_chr = downstream[downstream['Chr'] == chrm]
        sites = pandas.Series(['Intergenic' for i in range(chrSizes[chrm])])

        for i, r in plus_minus_ann_chr.iterrows():
            for j in range(r['+1 nucleosome'] - 74, r['+1 nucleosome'] + 74):
                if sites[j] == 'Intergenic': sites[j] = '+1_nuc_only'
                else: sites[j] = '+1_-1_nuc_both'

        for i, r in plus_minus_ann_chr.iterrows():
            for j in range(r['-1 nucleosome'] - 74, r['-1 nucleosome'] + 74):
                if sites[j] == 'Intergenic': sites[j] = '-1_nuc_only'
                else: sites[j] = '+1_-1_nuc_both'

        for i, r in promoter_chr.iterrows():
            for j in range(int(r['Promoter2k_start'] - 1), min(chrSizes[chrm], int(r['Promoter2k_end']))):
                if sites[j] == 'Intergenic': sites[j] = 'Promoter'

        for i, r in plus_minus_ann_chr.iterrows():
            if r['Strand'] == 1:
                orf_start = min(r['TSS'], r['ORF Start'])
                orf_end = max(r['TTS'], r['ORF End'])
            else:
                orf_start = min(r['TTS'], r['ORF End'])
                orf_end = max(r['TTS'], r['ORF Start'])
            for j in range(orf_start - 1, orf_end):
                if sites[j] == 'Intergenic': sites[j] = 'ORF_transcript'

        for i, r in downstream_chr.iterrows():
            for j in range(int(r['Downstream500_start'] - 1), min(chrSizes[chrm], int(r['Downstream500_end']))):
                if sites[j] == 'Intergenic': sites[j] = 'Downstream'
                
        for k in d:
            d[k] += len(sites[sites == k])

    print(d)

