######################################################
# Compute parameters from data
######################################################
from robocop import robocop
from robocop.utils.parameters import computeLinkers, computeMNaseBackground, computeMNaseTFPhisMus, computeMNaseNucMusPhis, computeMNaseNucOneMusPhis
import numpy as np
import pickle
from robocop.utils import concentration_probability_conversion
from Bio import SeqIO

def computeBackground(fastaFile):
    """
    Calculate the background distribution
    """
    bg = np.zeros((4, 1))
    fastaSeq = list(SeqIO.parse(open(fastaFile), 'fasta'))
    for fs in fastaSeq:
        seq = fs.seq
        bg[0, 0] += seq.count("A") + seq.count("a")
        bg[1, 0] += seq.count("C") + seq.count("c")
        bg[2, 0] += seq.count("G") + seq.count("g")
        bg[3, 0] += seq.count("T") + seq.count("t")
    bg = bg/np.sum(bg)
    # print("Calculated bg:", bg)
    return bg

def computeUnknown(bgmotif):
    """
    Calculate unknown motif from background motif
    """
    uk = np.zeros((4, 10))
    for i in range(4):
        uk[i, :] = bgmotif[i, 0]
    return uk

# Kd of most optimal sequence according to pwm
def calculateKD(pwm, k):
    score = 0
    for i in range(len(pwm[k]['matrix'][0])):
        idx = np.argmax(pwm[k]['matrix'][:, i])
        score += np.log10(pwm['background']['matrix'][idx]) - np.log10(pwm[k]['matrix'][idx, i])
    return 10**score

def getDBFconc(nucFile, pwmFile):

    pwm = pickle.load(open(pwmFile, "rb"), encoding = 'latin1')
    # remove secondary motif

    #secondaryMotifs = [x for x in list(pwm.keys()) if "secondary" in x]
    #for motif in secondaryMotifs: pwm.pop(motif)
    # print("TFs:", sorted(list(pwm.keys())))
    
    pwm['background'] = {"matrix": computeBackground(nucFile)}
    pwm['unknown'] = {"matrix": computeUnknown(pwm['background']['matrix'])}
    # # remove at/gc rich motifs
    # atgc = ['Azf1', 'Nhp6a', 'Nhp6b', 'Pho2', 'Sfp1', 'Sig1', 'Smp1', 'Spt15', 'Stb3', 'Sum1', 'Yox1', 'Asg1', 'Cat8', 'Gal4', 'Hal9', 'Msn2', 'Nhp10', 'Pdr1', 'Put3', 'Rei1', 'Rpn4', 'Rsc30', 'Rsc3', 'Sip4', 'Skn7', 'Stp1', 'Stp2', 'Swi5', 'Uga3', 'Yer184c', 'Yll054c']
    # atgcmotifs = list(filter(lambda x: x.split("_")[0] in atgc, pwm.keys()))
    # for motif in atgcmotifs: pwm.pop(motif)

    # # remove unknown
    # pwm.pop('unknown_TF')

    dbf_conc = [(k, calculateKD(pwm, k)) for k in list(pwm.keys())]
    # dbf_conc.append(('background', 1.0))
    dbf_conc.append(('nucleosome', 35))
    dbf_conc = dict(dbf_conc)

    # print("DBF conc.", dbf_conc)

    print("Number of TFs in my list:", len(list(dbf_conc.keys())) - 2)

    # convert concentration to probability
    dbf_conc = concentration_probability_conversion.convert_to_prob(dbf_conc, pwm)
    dbf_conc_sum = sum(dbf_conc.values())
    for k in list(dbf_conc.keys()):
        dbf_conc[k] = dbf_conc[k]/dbf_conc_sum
    return dbf_conc, pwm

# parameterize MNase-seq midpoint counts using negative binomial distribution
def getParamsMNase(mnaseFile, nucFile, tfFile, fragRange, tech = "MNase"):
    # fragRange = [(127, 187), (0, 80)]
    offset = 4 if tech == "ATAC" else 0
    if mnaseFile:
        # get linker coordinates from nucleosome file
        segments = computeLinkers(nucFile)
        # compute NB parameters for counts in linker region
        otherShort = computeMNaseBackground(mnaseFile, segments, fragRange[1], offset) # {'mu': 0.12367589449681342, 'phi': 0.12745604889725998}
        otherLong = computeMNaseBackground(mnaseFile, segments, fragRange[0], offset) # {'mu': 0.49634271906346195, 'phi': 0.25951075063267554}
        mus, phis = computeMNaseNucMusPhis(mnaseFile, nucFile, fragRange[0], offset)
        nucLong = {}
        # nucLong['mu'] = np.load('nucLongMu.npy')
        # nucLong['phi'] = 0.4644386359048939
        nucLong['mu'] = mus
        nucLong['phi'] = np.mean(phis)
        nucLong['scale'] = 1
        #np.save("nucLongMu", mus)
        nucShort = {'mu': otherShort['mu'], 'phi': otherShort['phi']}
        nucShort['scale'] = np.ones(147)

        # fit NB to counts in TF sites 
        tfShort = computeMNaseTFPhisMus(mnaseFile, tfFile, fragRange[1], None, offset) # {'mu': 2.05080615363017, 'phi': 0.4957510592300852}
        print("Computed TF short:", tfShort)
        # long count distribution is same as background
        tfLong = {'mu': otherLong['mu'], 'phi': otherLong['phi']}
        # newtfLong = computeMNaseTFPhisMus(mnaseFile, tfFile, fragRange[1], None)
        # newnucShort = computeMNaseNucOneMusPhis(mnaseFile, nucFile, fragRange[1])
        # print("oldShort:", otherShort)
        # print("new nuc short:", newnucShort)
        # print("oldLong:", otherLong)
        # print("new tf long:", newtfLong)

        # tfLong = newtfLong
        # mus, phis = newnucShort
        # nucShort = {'mu': mus, 'phi': phis}
        # nucShort['scale'] = np.ones(147)
        
        # exit(0)
        
        mnaseParams = {'nucLong': nucLong, 'nucShort': nucShort, 'otherLong': otherLong, 'otherShort': otherShort, 'tfLong': tfLong, 'tfShort': tfShort}
        return mnaseParams
