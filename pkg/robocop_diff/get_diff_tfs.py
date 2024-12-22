import matplotlib.pyplot as plt
import seaborn
import pandas
import numpy as np
import h5py
import os
from scipy import sparse
import pickle
from configparser import ConfigParser
# from configparser import SafeConfigParser
import scipy.stats as st

def get_sparse(f, k):
    g = f[k]
    v_sparse = sparse.csr_matrix((g['data'][:],g['indices'][:], g['indptr'][:]), g.attrs['shape'])
    return v_sparse

def get_sparse_todense(f, k):
    v_dense = np.array(get_sparse(f, k).todense())
    if v_dense.shape[0]==1: v_dense = v_dense[0]
    return v_dense


def get_sparse_tf_scores_df(filename):
    f = h5py.File(filename, 'r')
    chrs = f.keys()
    dfs = []
    for c in chrs:
        scores = get_sparse_todense(f, c + '/score')
        df_c = pandas.DataFrame(columns=['chr', 'start', 'end', 'width', 'score'])
        df_c['score'] = scores
        df_c['chr'] = c
        df_c['start'] = f[c+'/start'][:]
        # df_c['start'] = np.arange(1, scores.shape[0]+1).astype(int)
        df_c['width'] = f[c].attrs['width']
        df_c['end'] = df_c['start'] + df_c['width']
        dfs.append(df_c)

    df = pandas.concat(dfs, ignore_index=True)
    return df

def get_sites(dirnames, outdir, tf):
    if os.path.isfile(outdir + '/tmpDir/diff_tf_' + tf + '.csv'):
        tf_sites = pandas.read_csv(outdir + '/tmpDir/diff_tf_' + tf + '.csv', sep='\t')
        return tf_sites
    curr_pos = 'A'
    score_cols = []
    for d in dirnames:
        # tf_d = pandas.read_hdf(d + '/RoboCOP_outputs/' + tf + '.h5', mode='r', key='df')
        tf_d = get_sparse_tf_scores_df(d + '/RoboCOP_outputs/' + tf + '.h5')
        print(list(tf_d))
        tf_d = tf_d.rename(columns={'score': 'score' + curr_pos})
        # tf_d = tf_d.drop(columns=['index', 'width'])
        tf_d = tf_d.drop(columns=['width'])
        score_cols.append('score' + curr_pos)
        if curr_pos == 'A':
            tf_sites = tf_d
        else:
            tf_sites = tf_sites.merge(tf_d, on=['chr', 'start', 'end'], how='outer') 
        curr_pos = chr(ord(curr_pos) + 1)

    # tf_sites = tf_sites[~np.all(tf_sites[score_cols] > 0.1, axis=1)]
    tf_sites = tf_sites[np.any(tf_sites[score_cols] > 0.1, axis=1)]
    print("tf sites > 0.1:")
    print(tf_sites['start'].max(), tf_sites['end'].max())
    tf_sites['TF'] = [tf for _ in range(len(tf_sites))]
    tf_sites.to_csv(outdir+'/tmpDir/diff_tf_' + tf + '.csv', sep='\t', index=False)
    return tf_sites

def get_tf_sites(dirnames):
    config = ConfigParser() # SafeConfigParser()
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
    os.makedirs(outdir + 'tmpDir/', exist_ok=True)
    if os.path.isfile(outdir + 'diff_tf.csv'):
        all_tf_sites = pandas.read_csv(outdir + 'diff_tf.csv', sep='\t')
        return all_tf_sites
    
    # config = SafeConfigParser()
    config = ConfigParser()
    config.read(dirnames[0] + '/config.ini')
    hmmconfigfile = config.get("main", "trainDir") + "/HMMconfig.pkl"
    hmmconfig = pickle.load(open(hmmconfigfile, "rb"), encoding = "latin1")
    tfmotifs = hmmconfig["tfs"]
    tfs = []
    for i in tfmotifs: tfs.append((i.split("_")[0]).upper())
    tfs = list(set(tfs))
    tfs = list(filter(lambda x: x != "UNKNOWN" and x != "BACKGROUND", tfs))
    flag = 0
    tf_counter = 0
    for tf in tfs:
        tf_counter += 1
        print("TF iter:", tf_counter)
        print(tf)
        tf_sites = get_sites(dirnames, outdir, tf)
        if flag == 0:
            all_tf_sites = tf_sites
            flag = 1
        else:
            all_tf_sites = all_tf_sites.append(tf_sites, ignore_index=True)
    all_tf_sites.to_csv(outdir + 'diff_tf.csv', sep='\t', index=False)
    return all_tf_sites

def get_tf_gene_promoter(dirnames, outdir, plus_minus_ann):
    all_tf_sites = get_tf_diff(dirnames, outdir)
    score_cols = list(filter(lambda x: x.startswith('score'), all_tf_sites.columns))

    all_tf_sites = all_tf_sites[~np.all(all_tf_sites[score_cols] > 0.1, axis=1)]
    all_tf_sites = all_tf_sites.reset_index(drop=True)

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
    return all_tf_sites
    
def get_tf_clusters(gene_df, dirnames, outdir, plus_minus_ann):

    # plus_minus_ann = pandas.read_csv(plus_minus_ann_file, sep=',')
    tf_sites = get_tf_gene_promoter(dirnames, outdir, plus_minus_ann) 
    tf_in_genes = pandas.DataFrame(columns=['TF', 'cluster', 'cluster_size', 'n_tf_genes'])
    tf_sites = tf_sites[~tf_sites['gene'].isna()]
    tf_sites['gene'] = tf_sites['gene'].str.split(',')
    tf_sites = tf_sites.explode('gene')
    for c in sorted(list(set(gene_df['cluster']))):
        genes = list(gene_df[gene_df['cluster'] == c].index)
        gene_len = len(genes)
        for tf in sorted(list(set(tf_sites['TF']))):

            tf_genes = tf_sites[(tf_sites['TF'] == tf) & (tf_sites['gene'].isin(genes))]['gene'].unique()
            tf_in_genes = tf_in_genes.append({'TF': tf, 'cluster': c, 'cluster_size': gene_len, 'n_tf_genes': len(tf_genes)}, ignore_index=True)

    tf_in_genes['gene_percentage'] = tf_in_genes['n_tf_genes'] / tf_in_genes['cluster_size'] * 100

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

    return tf_wide


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def plot_tf_diff(gene_df, dirnames, plus_minus_ann, outdir, k):
    tfs = ['REB1', 'STP4', 'SWI4', 'RAP1', 'PBF1', 'PBF2', 'STP1', 'DAL82', 'RIM101', 'MET31', 'RTG3', 'SFL1', 'YPR196W', 'GAL4', 'GCR1', 'YKL222C', 'ABF1', 'ASG1', 'YBG033W']
    tf_sites = get_tf_gene_promoter(dirnames, outdir, plus_minus_ann) # pandas.read_csv('/usr/xtmp/sneha/tmpDir/all_diff_tfs_genes_subset.csv', sep='\t')
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

    for c in range(k):
        arr = tf_sites[tf_sites['cluster']==c][['score_diff_B', 'score_diff_C', 'score_diff_D', 'score_diff_E']].values
        arr = np.ravel(arr)
        arr = arr[~np.isnan(arr)]
        seaborn.distplot(arr, hist=False, ax=ax[1], label='G'+str(c+1))
    ax[1].legend()
    ax[0].set_title('All differences')
    ax[1].set_title('Cluster specific differences')
    # plt.savefig('/usr/xtmp/sneha/tmpDir/tmp.pdf')
    plt.show()
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
    # plt.savefig('/usr/xtmp/sneha/tmpDir/tmp_B.png')
    plt.show()
    plt.close()
    seaborn.clustermap(h_df_C.fillna(0), xticklabels=1, yticklabels=1, cmap='bwr', figsize=(10, 20), col_cluster=False, vmin=-0.4, vmax=0.4)
    # plt.savefig('/usr/xtmp/sneha/tmpDir/tmp_C.png')
    plt.show()
    plt.close()
    seaborn.clustermap(h_df_D.fillna(0), xticklabels=1, yticklabels=1, cmap='bwr', figsize=(10, 20), col_cluster=False, vmin=-0.4, vmax=0.4)
    # plt.savefig('/usr/xtmp/sneha/tmpDir/tmp_D.png')
    plt.show()
    plt.close()
    seaborn.clustermap(h_df_E.fillna(0), xticklabels=1, yticklabels=1, cmap='bwr', figsize=(10, 20), col_cluster=False, vmin=-0.4, vmax=0.4)
    # plt.savefig('/usr/xtmp/sneha/tmpDir/tmp_E.png')
    plt.show()
    plt.close()
