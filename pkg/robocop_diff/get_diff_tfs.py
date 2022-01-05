
import matplotlib.pyplot as plt
import seaborn
import pandas
import numpy as np
import h5py
import os
import pickle
from configparser import SafeConfigParser
import scipy.stats as st

def get_sites(dirnames, tf):
    if os.path.isfile('/usr/xtmp/sneha/tmpDir/diff_tf_' + tf + '.csv'):
        tf_sites = pandas.read_csv('/usr/xtmp/sneha/tmpDir/diff_tf_' + tf + '.csv', sep='\t')
        return tf_sites
    curr_pos = 'A'
    score_cols = []
    for d in dirnames:
        tf_d = pandas.read_hdf(d + '/RoboCOP_outputs/' + tf + '.h5', mode='r', key='df')
        print(list(tf_d))
        tf_d = tf_d.rename(columns={'score': 'score' + curr_pos})
        tf_d = tf_d.drop(columns=['index', 'width'])
        score_cols.append('score' + curr_pos)
        if curr_pos == 'A':
            tf_sites = tf_d
        else:
            tf_sites = tf_sites.merge(tf_d, on=['chr', 'start', 'end'], how='outer') 
        curr_pos = chr(ord(curr_pos) + 1)

    # tf_sites = tf_sites[~np.all(tf_sites[score_cols] > 0.1, axis=1)]
    tf_sites = tf_sites[np.any(tf_sites[score_cols] > 0.1, axis=1)]
    tf_sites['TF'] = [tf for _ in range(len(tf_sites))]
    print(tf_sites)
    tf_sites.to_csv('/usr/xtmp/sneha/tmpDir/diff_tf_' + tf + '.csv', sep='\t', index=False)
    return tf_sites

def get_tf_sites(dirnames):
    config = SafeConfigParser()
    config.read(dirnames[0] + '/config.ini')
    hmmconfigfile = config.get("main", "trainDir") + "/HMMconfig.pkl"
    hmmconfig = pickle.load(open(hmmconfigfile, "rb"), encoding = "latin1")
    tfmotifs = hmmconfig["tfs"]
    tfs = []
    for i in tfmotifs:
        tfs.append((i.split("_")[0]).upper())
    tfs = list(set(tfs))
    tfs = list(filter(lambda x: x != "UNKNOWN" and x != "BACKGROUND", tfs))
    print(tfs)
    for tf in tfs:
        print(tf)
        get_sites(dirnames, tf)

def get_tf_diff(dirnames, outdir):
    outdir = outdir + '/' if outdir[-1] != '/' else outdir 
    if os.path.isfile(outdir + 'diff_tf.csv'):
        all_tf_sites = pandas.read_csv(outdir + 'diff_tf.csv', sep='\t')
        return all_tf_sites
    
    config = SafeConfigParser()
    config.read(dirnames[0] + '/config.ini')
    hmmconfigfile = config.get("main", "trainDir") + "/HMMconfig.pkl"
    hmmconfig = pickle.load(open(hmmconfigfile, "rb"), encoding = "latin1")
    tfmotifs = hmmconfig["tfs"]
    tfs = []
    for i in tfmotifs: tfs.append((i.split("_")[0]).upper())
    tfs = list(set(tfs))
    tfs = list(filter(lambda x: x != "UNKNOWN" and x != "BACKGROUND", tfs))
    flag = 0
    
    for tf in tfs:
        print(tf)
        tf_sites = get_sites(dirnames, tf)
        if flag == 0:
            all_tf_sites = tf_sites
            flag = 1
        else:
            all_tf_sites = all_tf_sites.append(tf_sites, ignore_index=True)
    all_tf_sites.to_csv(outdir + 'diff_tf.csv', sep='\t', index=False)
    return all_tf_sites

def get_tf_gene_promoter(dirnames, outdir):
    '''
    if os.path.isfile('/usr/xtmp/sneha/tmpDir/all_diff_tfs_genes.csv'):
        all_tf_sites = pandas.read_csv('/usr/xtmp/sneha/tmpDir/all_diff_tfs_genes.csv', sep='\t')
        return all_tf_sites
    '''
    all_tf_sites = get_tf_diff(dirnames, outdir)
    score_cols = list(filter(lambda x: x.startswith('score'), all_tf_sites.columns))

    all_tf_sites = all_tf_sites[~np.all(all_tf_sites[score_cols] > 0.1, axis=1)]
    all_tf_sites = all_tf_sites.reset_index(drop=True)
    
    plus_minus_ann = pandas.read_csv('/usr/xtmp/sneha/Chereji_2018/13059_2018_1398_MOESM2_ESM.csv', sep=',')
    plus_minus_ann['Promoter_start'] = [r['-1 nucleosome'] - 73 if r['Strand'] == 1 else r['+1 nucleosome'] - 73 for i, r in plus_minus_ann.iterrows()]
    plus_minus_ann['Promoter_end'] = [r['+1 nucleosome'] + 73 if r['Strand'] == 1 else r['-1 nucleosome'] + 73 for i, r in plus_minus_ann.iterrows()]
    count_df = pandas.DataFrame(columns=['tf', 'total', 'promoter_count'])
    tfs = sorted(list(set(all_tf_sites['TF'])))
    genes = ['' for i in range(len(all_tf_sites))]
    c = dict([(tf, 0) for tf in tfs])

    for i, r in all_tf_sites.iterrows():
        pm_tf = plus_minus_ann[(plus_minus_ann['Promoter_start'] <= r['end']) & (plus_minus_ann['Promoter_end'] >= r['start']) & (plus_minus_ann['Chr'] == r['chr'])]
        if pm_tf.empty: continue
        c[r['TF']] += 1
        genes[i] = ','.join(pm_tf['ORF'])
    all_tf_sites['gene'] = genes
    # all_tf_sites.to_csv('/usr/xtmp/sneha/tmpDir/all_diff_tfs_genes.csv', sep='\t', index=False)
    return all_tf_sites
    '''
    for tf in tfs:
          count_df = count_df.append({'tf': tf, 'total': len(tf_sites[tf_sites['TF'] == tf]), 'promoter_count': c[tf]}, ignore_index=True)

    percentage = np.zeros(len(count_df))
    percentage[count_df['total'] > 0] = count_df[count_df['total'] > 0]['promoter_count'] / count_df[count_df['total'] > 0]['total'] * 100
    count_df['percentage'] = percentage
    count_df.to_csv('/usr/xtmp/sneha/tmpDir/all_diff_tfs_genes_counts.csv', sep='\t', index=False)
    return count_df
    exit(0)
    '''
    '''
    count_df = pandas.read_csv('/usr/xtmp/sneha/tmpDir/all_diff_tfs_genes_counts_subset.csv', sep='\t')
    fig, ax = plt.subplots(1, 3, figsize=(12, 19))
    seaborn.barplot(y='tf', x='total', data=count_df, ax=ax[0])
    seaborn.barplot(y='tf', x='promoter_count', data=count_df, ax=ax[1])
    seaborn.barplot(y='tf', x='percentage', data=count_df, ax=ax[2])
    '''
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)
    ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=90)
    '''
    plt.savefig('/usr/xtmp/sneha/tmpDir/tmp.png')
    plt.close()
    exit(0)
    '''
    
def get_tf_clusters(gene_df, dirnames, outdir):
    tf_sites = get_tf_gene_promoter(dirnames, outdir) # pandas.read_csv('/usr/xtmp/sneha/tmpDir/all_diff_tfs_genes_subset.csv', sep='\t')
    tf_in_genes = pandas.DataFrame(columns=['TF', 'cluster', 'cluster_size', 'n_tf_genes'])
    tf_sites = tf_sites[~tf_sites['gene'].isna()]
    tf_sites['gene'] = tf_sites['gene'].str.split(',')
    tf_sites = tf_sites.explode('gene')
    for c in sorted(list(set(gene_df['cluster']))):
        genes = list(gene_df[gene_df['cluster'] == c].index)
        gene_len = len(genes)
        for tf in sorted(list(set(tf_sites['TF']))):
            # if c == 7 and tf == 'MET31':
            #     p_tf = tf_sites[(tf_sites['TF'] == tf) & (tf_sites['gene'].isin(genes))]['gene'].unique()
            #     print(p_tf)
            #     exit(0)
            '''
            if c == 0: # and tf == 'ABF1':
                p_tf = tf_sites[(tf_sites['gene'].isin(genes))]['gene'].unique()
                for g in p_tf:
                    t_s = tf_sites[tf_sites['gene'] == g]
                    g_df = gene_df.loc[g]
                    p1s = g_df[['p1_shift_AB', 'p1_shift_BC', 'p1_shift_CD', 'p1_shift_DE']].values
                    m1s = g_df[['m1_shift_AB', 'm1_shift_BC', 'm1_shift_CD', 'm1_shift_DE']].values
                    if np.any(p1s < -20) and np.any(m1s > 20):
                        print(g, len(t_s), p1s, m1s)
                print(sorted(list(p_tf)))
                exit(0)
            '''

            if c == 7: # and tf == 'ABF1':
                p_tf = tf_sites[(tf_sites['gene'].isin(genes))]['gene'].unique()
                for g in p_tf:
                    t_s = tf_sites[tf_sites['gene'] == g]
                    g_df = gene_df.loc[g]
                    p1s = g_df[['p1_shift_AB', 'p1_shift_BC', 'p1_shift_CD', 'p1_shift_DE']].values
                    m1s = g_df[['m1_shift_AB', 'm1_shift_BC', 'm1_shift_CD', 'm1_shift_DE']].values
                    if np.any(p1s > 20) and np.any(m1s < -20):
                        print(g, len(t_s), p1s, m1s)
                print(sorted(list(p_tf)))
                exit(0)

            print(c, tf)
            tf_genes = tf_sites[(tf_sites['TF'] == tf) & (tf_sites['gene'].isin(genes))]['gene'].unique()
            # print(tf_genes)
            # exit(0)
            # tf_genes = sum([x.split(',') for x in tf_sites[(tf_sites['tf'] == tf) & (~tf_sites['gene'].isna())]['gene']], [])
            # tf_genes = list(set(tf_genes))
            tf_in_genes = tf_in_genes.append({'TF': tf, 'cluster': c, 'cluster_size': gene_len, 'n_tf_genes': len(tf_genes)}, ignore_index=True)


    print(tf_in_genes)
    
    tf_in_genes['gene_percentage'] = tf_in_genes['n_tf_genes'] / tf_in_genes['cluster_size'] * 100

    '''
    seaborn.barplot(x='tf', y='gene_percentage', hue='cluster', data=tf_in_genes)
    plt.savefig('/usr/xtmp/sneha/tmpDir/tmp.png')
    plt.close()
    '''
    
    tf_names = sorted(list(set(tf_in_genes['TF'])))
    clusters = sorted(list(set(tf_in_genes['cluster'])))
    tf_wide =  np.zeros((len(set(tf_in_genes['TF'])), len(set(tf_in_genes['cluster']))))

    for i in clusters:
        tf_in_genes_subset = tf_in_genes[tf_in_genes['cluster'] == i].sort_values(by='TF')
        tf_wide[:, i] = tf_in_genes_subset['gene_percentage'].values

    tf_wide = pandas.DataFrame(tf_wide, columns=[c+1 for c in clusters], index=tf_names)
    tf_wide = tf_wide.loc[(tf_wide.sum(axis=1) != 0),]
    # select
    # not all > 10%
    tf_wide = tf_wide[~np.all(tf_wide > 10, axis=1)]
    # at least 1 > 10%
    tf_wide = tf_wide[np.any(tf_wide > 10, axis=1)]
    print("tf_wide")
    print(tf_wide)
    # print(np.sum(tf_wide.values, axis=0), np.sum(tf_wide.values, axis=1))
    # plt.figure(figsize=(6, 19))
    # seaborn.heatmap(tf_wide, cmap='Spectral_r', yticklabels=1, vmax=50)
    # seaborn.clustermap(tf_wide, cmap='Spectral_r', yticklabels=1, col_cluster=False, figsize=(8, 19), vmax=30) #, standard_scale=0)
    seaborn.clustermap(tf_wide, cmap='Spectral_r', yticklabels=1, col_cluster=False, figsize=(8, 19), z_score=0) # standard_scale=0)
    plt.savefig('/usr/xtmp/sneha/tmpDir/tmp.pdf')
    plt.close()

    tf_dirname = '/usr/xtmp/sneha/tmpDir/tf_diff_plots/'
    tf_sites_tf = tf_sites.drop(columns=['chr', 'start', 'end'])
    tf_sites_tf = tf_sites_tf.groupby(by=['gene', 'TF']).mean().reset_index()

    gene_tf_times = []
    for tf in tf_names: # ['MCM1']:
        tf_sites_one_tf = tf_sites_tf[tf_sites_tf['TF'] == tf].reset_index()
        tf_sites_one_tf = tf_sites_one_tf.drop(columns=['TF', 'index'])
        tf_sites_one_tf = tf_sites_one_tf.set_index('gene')
        genes_not_present = set(gene_df.index).difference(set(tf_sites_one_tf.index))
        col_names = sorted(list(filter(lambda x: x != 'gene', tf_sites_one_tf.columns)))
        genes_not_present_df = pandas.DataFrame(np.zeros((len(genes_not_present), len(col_names))), columns=col_names, index=genes_not_present)
        tf_sites_one_tf = tf_sites_one_tf.append(genes_not_present_df, ignore_index=False)
        tf_sites_one_tf = tf_sites_one_tf.loc[gene_df.index]
        
        plt.figure(figsize=(4, 19))
        ax = seaborn.heatmap(tf_sites_one_tf, cmap='Blues')
        ax.set_yticks([])

        csize = 0
        for i in range(len(set(gene_df['cluster']))):
            if csize > 0: ax.axhline(csize, linewidth=2, color='black')
            csize += len(gene_df[gene_df['cluster'] == i])
        
        plt.savefig(tf_dirname + tf + '.png')
        plt.close()
        gene_tf_times.append(tf_sites_one_tf.values)

    gene_tf_times = np.array(gene_tf_times)
    print(gene_tf_times.shape)
    # gene_tf_times = np.swapaxes(gene_tf_times, 1, 2)
    # print(gene_tf_times.shape)
    
    np.save('/usr/xtmp/sneha/tmpDir/gene_tf_probs', gene_tf_times)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def plot_tf_diff(gene_df, dirnames, outdir):
    tfs = ['REB1', 'STP4', 'SWI4', 'RAP1', 'PBF1', 'PBF2', 'STP1', 'DAL82', 'RIM101', 'MET31', 'RTG3', 'SFL1', 'YPR196W', 'GAL4', 'GCR1', 'YKL222C', 'ABF1', 'ASG1', 'YBG033W']
    tf_sites = get_tf_gene_promoter(dirnames, outdir) # pandas.read_csv('/usr/xtmp/sneha/tmpDir/all_diff_tfs_genes_subset.csv', sep='\t')
    tf_in_genes = pandas.DataFrame(columns=['TF', 'cluster', 'cluster_size', 'n_tf_genes'])
    tf_sites = tf_sites[~tf_sites['gene'].isna()]
    tf_sites['gene'] = tf_sites['gene'].str.split(',')
    tf_sites = tf_sites.explode('gene')
    tf_sites = tf_sites[~tf_sites['gene'].isna()]

    tf_sites = tf_sites.merge(gene_df, on='gene')
    tf_sites['score_diff_E'] = tf_sites['scoreE'] - tf_sites['scoreA']
    tf_sites['score_diff_D'] = tf_sites['scoreD'] - tf_sites['scoreA']
    tf_sites['score_diff_C'] = tf_sites['scoreC'] - tf_sites['scoreA']
    tf_sites['score_diff_B'] = tf_sites['scoreB'] - tf_sites['scoreA']

    # fig, ax = plt.subplots(4, 2, sharex=True, sharey=True, figsize=(10, 12))
    # row = 0
    # col = 0
    h_df_B = pandas.DataFrame(columns=sorted(list(set(tf_sites['TF']))))
    h_df_C = pandas.DataFrame(columns=sorted(list(set(tf_sites['TF']))))
    h_df_D = pandas.DataFrame(columns=sorted(list(set(tf_sites['TF']))))
    h_df_E = pandas.DataFrame(columns=sorted(list(set(tf_sites['TF']))))
    for c in sorted(list(set(tf_sites['cluster']))):
        
        tf_sites_counts = tf_sites[tf_sites['cluster'] == c]
        tf_sites_counts = tf_sites_counts[['TF']].groupby('TF').size().reset_index(name='counts').set_index('TF')

        tf_sites_c = tf_sites[tf_sites['cluster'] == c]
        tf_sites_c = tf_sites_c.groupby('TF')['score_diff_B'].sum().reset_index()
        tf_sites_c['score_diff_B'] = [r['score_diff_B'] / tf_sites_counts.loc[r['TF']]['counts'] for i, r in tf_sites_c.iterrows()]
        h_df_B = h_df_B.append(tf_sites_c.set_index('TF').T, ignore_index=True)
        
        tf_sites_c = tf_sites[tf_sites['cluster'] == c]
        tf_sites_c = tf_sites_c.groupby('TF')['score_diff_C'].sum().reset_index()
        tf_sites_c['score_diff_C'] = [r['score_diff_C'] / tf_sites_counts.loc[r['TF']]['counts'] for i, r in tf_sites_c.iterrows()]
        h_df_C = h_df_C.append(tf_sites_c.set_index('TF').T, ignore_index=True)

        tf_sites_c = tf_sites[tf_sites['cluster'] == c]
        tf_sites_c = tf_sites_c.groupby('TF')['score_diff_D'].sum().reset_index()
        tf_sites_c['score_diff_D'] = [r['score_diff_D'] / tf_sites_counts.loc[r['TF']]['counts'] for i, r in tf_sites_c.iterrows()]
        h_df_D = h_df_D.append(tf_sites_c.set_index('TF').T, ignore_index=True)

        tf_sites_c = tf_sites[tf_sites['cluster'] == c]
        tf_sites_c = tf_sites_c.groupby('TF')['score_diff_E'].sum().reset_index()
        tf_sites_c['score_diff_E'] = [r['score_diff_E'] / tf_sites_counts.loc[r['TF']]['counts'] for i, r in tf_sites_c.iterrows()]
        h_df_E = h_df_E.append(tf_sites_c.set_index('TF').T, ignore_index=True)

        
    #     tf_sites_c = tf_sites_c[tf_sites_c['TF'].isin(tfs)]
    #     seaborn.barplot(data=tf_sites_c, x='TF', y='score_diff', ax=ax[row][col], order=tfs)
    #     # ax[row][col].set_xticklabels(ax[row][col].get_xticks(), rotation = 90)
    #     ax[row][col].set_xticklabels(tfs, rotation = 90)
    #     ax[row][col].set_title("G" + str(c+1))
    #     col += 1
    #     if col == 2:
    #         col = 0
    #         row += 1

    # plt.savefig('/usr/xtmp/sneha/tmpDir/tmp.png')
    # plt.close()

    arr = tf_sites[['score_diff_B', 'score_diff_C', 'score_diff_D', 'score_diff_E']].values
    arr = np.ravel(arr)
    arr = arr[~np.isnan(arr)]
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 6))
    seaborn.distplot(arr, hist=False, ax=ax[0])
    sd = np.std(arr)
    lim1 = np.mean(arr) - 2*sd
    lim2 = np.mean(arr) + 2*sd
    print(lim1, lim2)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    # ax[0].axvline(lim1, ls='--')
    # ax[0].axvline(lim2, ls='--')
    print(q1, q3)
    ax[0].axvline(q1, ls='--', color='red')
    ax[0].axvline(q3, ls='--', color='red')

    for c in range(8):
        arr = tf_sites[tf_sites['cluster']==c][['score_diff_B', 'score_diff_C', 'score_diff_D', 'score_diff_E']].values
        arr = np.ravel(arr)
        arr = arr[~np.isnan(arr)]
        seaborn.distplot(arr, hist=False, ax=ax[1], label='G'+str(c+1))
    ax[1].legend()
    ax[0].set_title('All differences')
    ax[1].set_title('Cluster specific differences')
    plt.savefig('/usr/xtmp/sneha/tmpDir/tmp.pdf')
    plt.close()
    # exit(0)
    
    h_df_B = h_df_B.T
    h_df_B.columns = ['G' + str(c+1) for c in sorted(list(set(tf_sites['cluster'])))]
    h_df_C = h_df_C.T
    h_df_C.columns = ['G' + str(c+1) for c in sorted(list(set(tf_sites['cluster'])))]
    h_df_D = h_df_D.T
    h_df_D.columns = ['G' + str(c+1) for c in sorted(list(set(tf_sites['cluster'])))]
    h_df_E = h_df_E.T
    h_df_E.columns = ['G' + str(c+1) for c in sorted(list(set(tf_sites['cluster'])))]
    print(h_df_B)

    seaborn.clustermap(h_df_B.fillna(0), xticklabels=1, yticklabels=1, cmap='bwr', figsize=(10, 20), col_cluster=False, vmin=-0.4, vmax=0.4)
    plt.savefig('/usr/xtmp/sneha/tmpDir/tmp_B.png')
    plt.close()
    seaborn.clustermap(h_df_C.fillna(0), xticklabels=1, yticklabels=1, cmap='bwr', figsize=(10, 20), col_cluster=False, vmin=-0.4, vmax=0.4)
    plt.savefig('/usr/xtmp/sneha/tmpDir/tmp_C.png')
    plt.close()
    seaborn.clustermap(h_df_D.fillna(0), xticklabels=1, yticklabels=1, cmap='bwr', figsize=(10, 20), col_cluster=False, vmin=-0.4, vmax=0.4)
    plt.savefig('/usr/xtmp/sneha/tmpDir/tmp_D.png')
    plt.close()
    seaborn.clustermap(h_df_E.fillna(0), xticklabels=1, yticklabels=1, cmap='bwr', figsize=(10, 20), col_cluster=False, vmin=-0.4, vmax=0.4)
    plt.savefig('/usr/xtmp/sneha/tmpDir/tmp_E.png')
    plt.close()
