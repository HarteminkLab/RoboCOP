######################################################
# Compute parameters from data
######################################################
from robocop import robocop
from robocop.utils.parameters import computeLinkers, computeMNaseBackground, computeMNaseBackgroundKernel, computeMNaseTFPhisMus, computeMNaseTFPhisMusKernel, computeMNaseNucMusPhis, computeMNaseNucMusPhisKernel, computeMNaseNucOneMusPhis
import numpy as np
import math
import pickle
from robocop.utils import concentration_probability_conversion
from Bio import SeqIO

def computeBackground(fastaFile):
    """
    Calculate the background distribution
    """
    bg = np.zeros((5, 1))
    fastaSeq = list(SeqIO.parse(open(fastaFile), 'fasta'))
    for fs in fastaSeq:
        seq = fs.seq
        bg[0, 0] += seq.count("A") + seq.count("a")
        bg[1, 0] += seq.count("C") + seq.count("c")
        bg[2, 0] += seq.count("G") + seq.count("g")
        bg[3, 0] += seq.count("T") + seq.count("t")
    bg[:4,:] = bg[:4,:]/np.sum(bg[:4,:])
    return bg

def computeUnknown(bgmotif):
    """
    Calculate unknown motif from background motif
    """
    uk = np.zeros((5, 10))
    for i in range(4):
        uk[i, :] = bgmotif[i, 0]
    return uk

# Kd of most optimal sequence according to pwm
def calculateKD(pwm, k):
    score = 0
    for i in range(len(pwm[k][0])):
        idx = np.argmax(pwm[k][:, i])
        score += math.log10(np.ravel(pwm['background'])[idx]) - math.log10(pwm[k][idx, i])
    return 10**score

def getMotifsMEME(pwmFile):
    motifLines = open(pwmFile, "r").readlines()
    motifLines = filter(lambda x: x != "\n", motifLines)

    foundMotif = 0
    motifDict = {}
    
    for i in motifLines:
        l = i.strip()
        if l[:4] == "URL ": continue
        if l[:5] == "MOTIF":
            l = l.split()
            foundMotif = l[1]
            motifDict[foundMotif] = [[], [], [], [], []]
            continue
        if foundMotif != 0 and l[:6] != "letter":
            l = l.split()
            for j in range(4):
                motifDict[foundMotif][j] = motifDict[foundMotif][j] + [float(l[j])]
            # N nucleotide
            motifDict[foundMotif][4] = motifDict[foundMotif][4] + [0]
    
    for m in motifDict.keys():
        motifDict[m] = np.array(motifDict[m])

    print("Number of motifs:", len(motifDict))
    return motifDict

def getDBFconc(nucFile, pwmFile, outDir):

    pwm = getMotifsMEME(pwmFile) 

    pwm['background'] = computeBackground(nucFile)
    pwm['unknown'] = computeUnknown(pwm['background'])

    pickle.dump(pwm, open(outDir + "/pwm.p", 'wb'))
    dbf_conc = [(k, calculateKD(pwm, k)) for k in list(pwm.keys())]
    dbf_conc = dict(dbf_conc)
    dbf_conc['background'] = 1.0
    dbf_conc['nucleosome'] = 35
    
    # convert concentration to probability
    dbf_conc = concentration_probability_conversion.convert_to_prob(dbf_conc, pwm)
    dbf_conc_sum = sum(dbf_conc.values())
    for k in list(dbf_conc.keys()):
        dbf_conc[k] = dbf_conc[k]/dbf_conc_sum
    return dbf_conc, pwm

# parameterize MNase-seq midpoint counts using negative binomial distribution
def getParamsMNase(mnaseFile, nucFile, tfFile, fragRange, tmpDir, tech = "MNase"):
    offset = 4 if tech == "ATAC" else 0
    if mnaseFile:
        # get linker coordinates from nucleosome file
        # fit NB to counts in TF sites 
        tfShort = computeMNaseTFPhisMus(mnaseFile, tfFile, tmpDir, fragRange[1], None, offset)
        segments = computeLinkers(nucFile)
        # compute NB parameters for counts in linker region
        otherShort = computeMNaseBackground(mnaseFile, tmpDir, segments, fragRange[1], offset, 'short')
        otherLong = computeMNaseBackground(mnaseFile, tmpDir, segments, fragRange[0], offset, 'long')
        # long count distribution is same as background
        tfLong = {'mu': otherLong['mu'], 'phi': otherLong['phi']}

        mus, phis = computeMNaseNucMusPhis(mnaseFile, nucFile, tmpDir, fragRange[0], offset)
        nucLong = {}
        nucLong['mu'] = mus
        nucLong['phi'] = np.mean(phis)
        nucLong['scale'] = 1
        nucShort = {'mu': otherShort['mu'], 'phi': otherShort['phi']}
        nucShort['scale'] = np.ones(147)

        mnaseParams = {'nucLong': nucLong, 'nucShort': nucShort, 'otherLong': otherLong, 'otherShort': otherShort, 'tfLong': tfLong, 'tfShort': tfShort}
        return mnaseParams


# parameterize MNase-seq midpoint counts using negative binomial distribution
def getParamsMNaseKernel(nucFile, tfFile, tmpDir):
    # get linker coordinates from nucleosome file
    segments = computeLinkers(nucFile)
    # compute NB parameters for counts in linker region
    tfShort = computeMNaseTFPhisMusKernel(tmpDir, tfFile, "tf") 
    otherShort = computeMNaseBackgroundKernel(tmpDir, segments, "tf") 
    otherLong = computeMNaseBackgroundKernel(tmpDir, segments, "nuc")
    shape, rate = computeMNaseNucMusPhisKernel(tmpDir, nucFile, "nuc")

    nucLong = {}
    nucLong['shape'] = shape
    nucLong['rate'] = rate
    nucLong['scale'] = 1
    nucShort = {'shape': otherShort['shape']*np.ones(147), 'rate': otherShort['rate']*np.ones(147)}
    nucShort['scale'] = 1 # np.ones(147)
    tfLong = {'shape': otherLong['shape'], 'rate': otherLong['rate']}


    mnaseParams = {'nucLong': nucLong, 'nucShort': nucShort, 'otherLong': otherLong, 'otherShort': otherShort, 'tfLong': tfLong, 'tfShort': tfShort}
    return mnaseParams
