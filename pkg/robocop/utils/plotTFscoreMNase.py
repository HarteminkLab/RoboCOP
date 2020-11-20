#################################################################################################
# Plot RoboCOP predicted score with MNase-seq fragment aggregates.
#################################################################################################

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas
import matplotlib.patches as patches
#from Bio import SeqIO
import roman

# Convert nucleotides to 0, 1, 2, 3
def mapNucToInt(n):
    if n == 'A' or n == 'a':
        return 0
    elif n == 'C' or n == 'c':
        return 1
    elif n == 'G' or n == 'g':
        return 2
    else: return 3

# extract nucleotide sequence for given chromosome, start and stop
def getNucleotideSequence(fastaFile, chromosome, start = 0, stop = 0):
    fastaSeq = list(SeqIO.parse(open(fastaFile), 'fasta'))
    fastaSequence = {}
    sequenceLengths = {}
    for fs in fastaSeq:
        fastaSequence[roman.fromRoman(fs.name[3:])] = fs.seq
        sequenceLengths[roman.fromRoman(fs.name[3:])] = len(fs.seq)
    if stop > sequenceLengths[chromosome]:
        print("ERROR: Invalid stop position for chromosome", chromosome, start, stop, sequenceLengths[chromosome])
        exit(1)
    if start <= 0:
        print("ERROR: Invalid start position for chromosome", chromosome)
        exit(1)

    sequence = fastaSequence[chromosome][(start - 1) : stop]
    # print(sequence)
    sequence = np.asarray([mapNucToInt(x) for x in sequence])
    return sequence

def getTFscore(seq, pwm):
    seq = list(seq)
    score = np.zeros(len(seq))
    for i in range(len(pwm[1])//2, len(seq) - len(pwm[1])//2):
        s = 0
        for j in range(len(pwm[1])):
            s += np.log(pwm[seq[i - len(pwm[1])//2 + j], j])
        score[i] = s
    return score

def plotRegion(dirnames, idx, dshared, chrm, start, end, tfIdx, pwm, gene, cgene, sitesTF):
    dfs = []
    mns = []
    mnl = []
    #seq = getNucleotideSequence('/usr/project/compbio/sneha/genome/SacCer3.fa', int(chrm), start, end)
    #seqScore = getTFscore(seq, pwm)[2000:-2000]
    #if gene[-1] == 'C':
    #    seqScore = np.flip(seqScore)
    for d in dirnames:
        p = np.load(d + 'tmpDir/posterior_and_emission.idx' + str(idx) + '.npz')['posterior']
        p = p[2000:-2000, :]
        dfs.append(p)
        m = np.load(d + 'tmpDir/MNaseShort.idx' + str(idx) + '.npy')
        m = m[2000:-2000]
        mns.append(m)
        m = np.load(d + 'tmpDir/MNaseLong.idx' + str(idx) + '.npy')
        m = m[2000:-2000]
        mnl.append(m)
    indices = []
    for tfI in tfIdx:
        indices += range(dshared['tf_starts'][tfI], dshared['tf_starts'][tfI] + 2*dshared['tf_lens'][tfI])
    fig, ax = plt.subplots(len(dirnames), 1, sharex = True, figsize = (15, 25))
    for d in range(len(dirnames)):
        tbl = dfs[d]
        # tblcol = np.sum(tbl[:, [dshared['tf_starts'][tfIdx], dshared['tf_starts'][tfIdx] + dshared['tf_lens'][tfIdx]]], axis = 1)
        ##### print(tbl[:, indices])
        ##### tblcol = np.sum(tbl[:, indices], axis = 1)
        tblcol = []
        for tfI in tfIdx:
            # print(tbl[:, dshared['tf_starts'][tfI] : (dshared['tf_starts'][tfI] + 2*dshared['tf_lens'][tfI])])
            tblcol.append(np.sum(tbl[:, dshared['tf_starts'][tfI] : (dshared['tf_starts'][tfI] + 2*dshared['tf_lens'][tfI])], axis = 1))
        # for tfi in tfIdx[1:]:
            # print(tbl[:, dshared['tf_starts'][tfi] : (dshared['tf_starts'][tfi] + 2*dshared['tf_lens'][tfi])])
            # tblcol += np.sum(tbl[:, dshared['tf_starts'][tfi] : (dshared['tf_starts'][tfi] + 2*dshared['tf_lens'][tfi])], axis = 1)

        # print(tblcol, len(tblcol))
        # print(mns[d], len(mns[d]))
        # print(seqScore, len(seqScore))
        if gene[-1] == 'C':
            # print(tblcol)
            # tblcol = np.flip(tblcol)
            mns[d] = np.flip(mns[d])
            mnl[d] = np.flip(mnl[d])
        ax[d].plot(range(-500, 500), mns[d], color = 'blue')
        ax[d].set_xlim((-500, 500))
        axt = ax[d].twinx()
        # axt1 = ax[d].twinx()
        # print(len(tblcol))
        tblcol = [tblcol[0], tblcol[1] + tblcol[3], tblcol[2] + tblcol[4]]
        cols = ['red', 'green', 'magenta']
        tfs = ['Hcm1', 'Fkh1', 'Fkh2']
        ls = []
        labels = ['Hcm1', 'Fkh1', 'Fkh2']
        for s in sitesTF:
            m = (s[0] + s[1])/2
            if gene[-1] == 'C': s = (5000 - s[1], 5000 - s[0])
            # print((s[0], 0), s[1] - s[0], 1)
            rect = patches.Rectangle((s[0] - 2500, 0), s[1] - s[0], 1, color = 'gray', alpha = 0.3)
            axt.add_patch(rect)
            # axt.plot(range(-500, 500), tblcol, color = 'red')
        for tbc in range(len(tblcol)):
            l, = axt.plot(range(-500, 500), tblcol[tbc] if gene[-1] == 'W' else np.flip(tblcol[tbc]), color = cols[tbc], label = labels[tbc])
            # print(cgene, np.max(tblcol))
        axt.set_ylim((0, 1))
        axt.legend(ncol = 3, frameon = False)
        # axt1.plot(range(-500, 500), mnl[d], color = 'green')
            #axt.axvline(m - 2500, linewidth = 6)
        # axt.plot(range(-500, 500), seqScore, color = 'red')
        # print(np.max(tblcol), np.max(seqScore))
    ax[d].set_xlabel("Distance from TSS")
    plt.savefig('/usr/xtmp/sneha/tmpDir/' + cgene + '_Hcm1_Fkh1_Fkh2_separate.png')
    plt.close()

if __name__ == '__main__':
    coords = pandas.read_csv("/home/home3/sneha/Hartemink/MVCOMPETE/modifiedMVCOMPETE/pkg/unit_test/coordinates_cell_cycle_sacCer3.csv", sep = '\t')
    MNaseFiles = ["/usr/xtmp/sneha/data/MNase-seq/MacAlpine_cell_cyle/DMAH_64_82/DMAH_64_82_sacCer3.bam", "/usr/xtmp/sneha/data/MNase-seq/MacAlpine_cell_cyle/DMAH_65_83/DMAH_65_83_sacCer3.bam", "/usr/xtmp/sneha/data/MNase-seq/MacAlpine_cell_cyle/DMAH_66_84/DMAH_66_84_sacCer3.bam", "/usr/xtmp/sneha/data/MNase-seq/MacAlpine_cell_cyle/DMAH_67_85/DMAH_67_85_sacCer3.bam", "/usr/xtmp/sneha/data/MNase-seq/MacAlpine_cell_cyle/DMAH_68_86/DMAH_68_86_sacCer3.bam", "/usr/xtmp/sneha/data/MNase-seq/MacAlpine_cell_cyle/DMAH_69_87/DMAH_69_87_sacCer3.bam", "/usr/xtmp/sneha/data/MNase-seq/MacAlpine_cell_cyle/DMAH_71_88/DMAH_71_88_sacCer3.bam", "/usr/xtmp/sneha/data/MNase-seq/MacAlpine_cell_cyle/DMAH_72_89/DMAH_72_89_sacCer3.bam", "/usr/xtmp/sneha/data/MNase-seq/MacAlpine_cell_cyle/DMAH_73_90/DMAH_73_90_sacCer3.bam", "/usr/xtmp/sneha/data/MNase-seq/MacAlpine_cell_cyle/DMAH_74_91/DMAH_74_91_sacCer3.bam", "/usr/xtmp/sneha/data/MNase-seq/MacAlpine_cell_cyle/DMAH_75_92/DMAH_75_92_sacCer3.bam", "/usr/xtmp/sneha/data/MNase-seq/MacAlpine_cell_cyle/DMAH_76_93/DMAH_76_93_sacCer3.bam", "/usr/xtmp/sneha/data/MNase-seq/MacAlpine_cell_cyle/DMAH_77_94/DMAH_77_94_sacCer3.bam", "/usr/xtmp/sneha/data/MNase-seq/MacAlpine_cell_cyle/DMAH_78_95/DMAH_78_95_sacCer3.bam", "/usr/xtmp/sneha/data/MNase-seq/MacAlpine_cell_cyle/DMAH_79_96/DMAH_79_96_sacCer3.bam", "/usr/xtmp/sneha/data/MNase-seq/MacAlpine_cell_cyle/DMAH_80_97/DMAH_80_97_sacCer3.bam", "/usr/xtmp/sneha/data/MNase-seq/MacAlpine_cell_cyle/DMAH_81_98/DMAH_81_98_sacCer3.bam"]
    outDirs = ["/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_64_82_allMotifs_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_65_83_allMotifs_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_66_84_allMotifs_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_67_85_allMotifs_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_68_86_allMotifs_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_69_87_allMotifs_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_71_88_allMotifs_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_72_89_allMotifs_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_73_90_allMotifs_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_74_91_allMotifs_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_75_92_allMotifs_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_76_93_allMotifs_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_77_94_allMotifs_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_78_95_allMotifs_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_79_96_allMotifs_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_80_97_allMotifs_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_81_98_allMotifs_cell_cycle_genes/"]
    # outDirs = ["/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_64_82_shortFrag_100_nucFrag_136_196_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_66_84_shortFrag_100_nucFrag_136_196_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_68_86_shortFrag_100_nucFrag_136_196_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_71_88_shortFrag_100_nucFrag_136_196_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_73_90_shortFrag_100_nucFrag_136_196_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_75_92_shortFrag_100_nucFrag_136_196_cell_cycle_genes/"]
    # outDirs = ["/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_64_82_shortFrag_100_nucFrag_131_191_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_66_84_shortFrag_100_nucFrag_131_191_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_68_86_shortFrag_100_nucFrag_131_191_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_71_88_shortFrag_100_nucFrag_131_191_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_73_90_shortFrag_100_nucFrag_131_191_cell_cycle_genes/", "/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_75_92_shortFrag_100_nucFrag_131_191_cell_cycle_genes/"]
    hmmconfigfile = '/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_64_82_allMotifs_Chr4/HMMconfig.pkl'
    # hmmconfigfile = '/usr/xtmp/sneha/RoboCOP_cell_cycle_new/DMAH_64_82_shortFrag_100_nucFrag_136_196_Chr4/HMMconfig.pkl'
    dshared = pickle.load(open(hmmconfigfile, "rb"))
    indices = [459, 270, 460, 461] # YHP1, FKH1, FKH2, NDD1
    genes = ['YHP1', 'FKH1', 'FKH2', 'NDD1']
    pwm = pickle.load(open('/home/home3/sneha/Hartemink/MVCOMPETE/modifiedMVCOMPETE/pkg/mvcompete/pwm/pwm_Gordan_union_MacIsaac_withCXrap1.p', 'rb'), encoding = 'latin1')
    tfpwm = pwm['Hcm1_badis']['matrix']
    tfIdx = [list(dshared['tfs']).index('Hcm1_badis')]
    tfIdx.append(list(dshared['tfs']).index('Fkh1_zhu'))
    tfIdx.append(list(dshared['tfs']).index('Fkh1_zhu'))
    tfIdx.append(list(dshared['tfs']).index('FKH1'))
    tfIdx.append(list(dshared['tfs']).index('FKH2'))
    sites = [[(2619, 2626)], [(2554, 2561), (835, 842), (2532, 2539), (1277, 1284)], [(4750, 4757), (2510, 2517), (425, 432), (4803, 4810), (3783, 3790)], [(2635, 2642), (1794, 1801)]]
    for idx in range(len(indices)):
        idc = indices[idx]
        chrm = int(coords.iloc[idc]['chr'])
        start = int(coords.iloc[idc]['start'])
        end = int(coords.iloc[idc]['end'])
        gene = coords.iloc[idx]['gene']
        print(tfIdx)
        plotRegion(outDirs, idc, dshared, chrm, start, end, tfIdx, tfpwm, gene, genes[idx], sites[idx])
