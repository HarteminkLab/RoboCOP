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

    pwm['background'] = {"matrix": computeBackground(nucFile)}
    pwm['unknown'] = {"matrix": computeUnknown(pwm['background']['matrix'])}

    dbf_conc = [(k, calculateKD(pwm, k)) for k in list(pwm.keys())]
    dbf_conc = dict(dbf_conc)
    dbf_conc['background'] = 1.0
    dbf_conc['nucleosome'] = 35
    
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
        otherShort = computeMNaseBackground(mnaseFile, segments, fragRange[1], offset)
        otherLong = computeMNaseBackground(mnaseFile, segments, fragRange[0], offset)
        mus, phis = computeMNaseNucMusPhis(mnaseFile, nucFile, fragRange[0], offset)
        nucLong = {}
        nucLong['mu'] = mus
        nucLong['phi'] = np.mean(phis)
        nucLong['scale'] = 1
        nucShort = {'mu': otherShort['mu'], 'phi': otherShort['phi']}
        nucShort['scale'] = np.ones(147)

        # fit NB to counts in TF sites 
        tfShort = computeMNaseTFPhisMus(mnaseFile, tfFile, fragRange[1], None, offset)
        # long count distribution is same as background
        tfLong = {'mu': otherLong['mu'], 'phi': otherLong['phi']}
        mnaseParams = {'nucLong': nucLong, 'nucShort': nucShort, 'otherLong': otherLong, 'otherShort': otherShort, 'tfLong': tfLong, 'tfShort': tfShort}
        return mnaseParams
