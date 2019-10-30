######################################################
# Compute parameters from data
######################################################
from robocop import robocop
from robocop.utils.parameters import computeLinkers, computeMNaseBackground, computeMNaseTFPhisMus, computeMNaseNucMusPhis
import numpy as np
import pickle
from robocop.utils import concentration_probability_conversion

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
    secondaryMotifs = [x for x in list(pwm.keys()) if "secondary" in x]
    for motif in secondaryMotifs: pwm.pop(motif)
    dbf_conc = [(k, calculateKD(pwm, k)) for k in list(pwm.keys())]
    dbf_conc.append(('background', 1.0))
    dbf_conc.append(('nucleosome', 35))
    dbf_conc = dict(dbf_conc)
        
    print("Number of TFs in my list:", len(list(dbf_conc.keys())) - 2)

    # convert concentration to probability
    dbf_conc = concentration_probability_conversion.convert_to_prob(dbf_conc, pwm)
    dbf_conc_sum = sum(dbf_conc.values())
    for k in list(dbf_conc.keys()):
        dbf_conc[k] = dbf_conc[k]/dbf_conc_sum
    return dbf_conc

# parameterize MNase-seq midpoint counts using negative binomial distribution
def getParamsMNase(mnaseFile, nucFile, tfFile):
    fragRange = [(127, 187), (0, 80)]
    if mnaseFile:
        # get linker coordinates from nucleosome file
        segments = computeLinkers(nucFile)
        # compute NB parameters for counts in linker region
        otherShort = computeMNaseBackground(mnaseFile, segments, fragRange[1])
        otherLong = computeMNaseBackground(mnaseFile, segments, fragRange[0])
        mus, phis = computeMNaseNucMusPhis(mnaseFile, nucFile, fragRange[0])
        nucLong = {}
        nucLong['mu'] = mus
        nucLong['phi'] = np.mean(phis)
        nucLong['scale'] = 1

        nucShort = {'mu': otherShort['mu'], 'phi': otherShort['phi']}
        nucShort['scale'] = np.ones(147)

        # fit NB to counts in TF sites 
        tfShort = computeMNaseTFPhisMus(mnaseFile, tfFile, fragRange[1], None)
        # long count distribution is same as background
        tfLong = {'mu': otherLong['mu'], 'phi': otherLong['phi']}
        mnaseParams = {'nucLong': nucLong, 'nucShort': nucShort, 'otherLong': otherLong, 'otherShort': otherShort, 'tfLong': tfLong, 'tfShort': tfShort}
        return mnaseParams
