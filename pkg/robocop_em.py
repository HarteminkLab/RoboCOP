import robocop
from robocop.utils.readWriteOps import *
from robocop.utils.robocopExtras import *
import robocop.utils.parameterize as parameterize
import numpy as np
import pickle
import os
import re
import sys
from configparser import SafeConfigParser
import robocop.utils.getReads as getReads
import pandas
import gc
from multiprocessing import Pool, Manager

# create posterior table for each segment
def createInstances(tf_prob, dbf_conc, coords, pwmFile, cshared, tmpDir, nucleotide_sequence, mnaseParams, pool):
    segments = len(coords)
    manager = Manager()
    dshared = manager.dict()
    dshared["robocopC"] = cshared 
    robocop.createSharedDictionary(dshared, tf_prob, dbf_conc['background'], dbf_conc['nucleosome'], pwmFile, tmpDir, nucleotide_sequence)
    pool.map(createInstance, [(t, dshared) for t in range(segments)])
    if mnaseParams != None: list(map(updateMNaseEMMatNB, [(s, dshared, mnaseParams) for s in range(segments)]))
    pool.map(posterior_forward_backward_wrapper, [(t, dshared) for t in range(segments)])
    gc.collect()
    return dshared
    
def runROBOCOP_EM(coordFile, config, outDir, tmpDir, pool, dnaseFiles = ""):

    fragRangeLong = tuple([int(x) for x in re.findall(r'\d+', config.get("main", "fragRangeLong"))])
    fragRangeShort = tuple([int(x) for x in re.findall(r'\d+', config.get("main", "fragRangeShort"))])
    fragRange = (fragRangeLong, fragRangeShort)
    nucFile = config.get("main", "nucFile")
    mnaseFiles = config.get("main", "mnaseFile")
    cshared = config.get("main", "cshared")
    # chromosome segments in pandas data frame
    coords = pandas.read_csv(coordFile, sep = "\t")

    # dbf weights initlaized using KD values
    dbf_conc = parameterize.getDBFconc(nucFile, config.get("main", "pwmFile"))

    # read nucleotide sequence and return 1 if successful
    nucleotide_sequence = getReads.getNucSequence(nucFile, tmpDir, coords)
    # read MNase-seq midpoint counts for long and short fragments
    mnase_data_long, mnase_data_short = getReads.getMNase(mnaseFiles, tmpDir, coords, fragRange)
    
    # make t copies of each tf_prob for every timepoint -- only 1 timepoint
    timepoints = len(coords)
    segments = len(coords)
    tf_prob = dict()
    threshold = 0
    thresholds = []
    # determine max prob for TFs to perform constrained EM optimization
    for i in list(dbf_conc.keys()):
        if i != 'nucleosome' and i != 'background':
            tf_prob[i] = dbf_conc[i] 
            if threshold < tf_prob[i] and i != 'unknown_TF':
                threshold = tf_prob[i][0]
            if i != 'unknown_TF': thresholds.append(tf_prob[i][0])
    threshold = np.mean(thresholds) + 2*np.std(thresholds)
    print("Threshold is:", threshold)
    # get MNase-seq count parameters
    if mnaseFiles: 
        mnaseParams = parameterize.getParamsMNase(mnaseFiles, config.get("main", "nucleosomeFile"), config.get("main", "tfFile"))
    else:
        mnaseParams = None
    
    # create shared dictionary for all segments and build HMM transition matrix
    dshared = createInstances(tf_prob, dbf_conc, coords, config.get("main", "pwmFile"), cshared, tmpDir, nucleotide_sequence, mnaseParams, pool)
    fLike = open(outDir + '/likelihood.txt', 'w')
    likelihood = getLogLikelihood(segments, dshared['tmpDir'])
    fLike.write(str(likelihood) + '\n')
    print("Likelihood before EM:", getLogLikelihood(segments, dshared['tmpDir']))
    iterations = 10 #150 #200 #40
    countMNase = 0
    lfMNaseLong = 1
    lfMNaseShort = 1
    lfMNaseScale = 1
    for i in range(iterations):
        # Baum-Welch on transition probabilities
        background_prob, _tf_prob, nucleosome_prob = update_transition_probs(dshared, segments, tmpDir, threshold)
        tf_prob = np.array([_tf_prob[_] for _ in np.array(sorted(_tf_prob.keys()), order = 'c')])
        robocop.set_transition(dshared, tf_prob, background_prob, nucleosome_prob)
        robocop.set_initial_probs(dshared)

        # posterior decoding with updated transition probabilities
        pool.map(setValuesPosterior, [(t, dshared, tf_prob, background_prob, nucleosome_prob, tmpDir) for t in range(segments)])
        likelihood = getLogLikelihood(segments, dshared['tmpDir'])
        print("Likelihood in iter:", i, getLogLikelihood(segments, dshared['tmpDir']))
        fLike.write(str(likelihood) + '\n')
    # printPosterior(segments, dshared, tmpDir, outDir)
    
    likelihood = getLogLikelihood(segments, dshared['tmpDir'])
    print("Likelihood after EM:", likelihood) #x.get_log_likelihood()
    fLike.write(str(likelihood) + '\n')
    fLike.close()
    
    #create new dshared
    dsharedNew = {}
    for k in list(dshared.keys()):
        dsharedNew[k] = dshared[k]
    with open(outDir + "/HMMconfig.pkl", 'wb') as writeFile:
        pickle.dump(dsharedNew, writeFile, pickle.HIGHEST_PROTOCOL)
    if mnaseFiles != "":
        with open(outDir + "/negParamsMNase.pkl", 'wb') as writeFile:
            pickle.dump(mnaseParams, writeFile, pickle.HIGHEST_PROTOCOL)
    return likelihood

if __name__ == '__main__':
        if len(sys.argv) != 4:
                print("Usage: python em_test_mispoints_onePhi_mnaseMNase.py <coordinate file> <config file> <outputDir>")
                exit(1)
        coordFile = sys.argv[1]
        configFile = sys.argv[2]
        outDir = (sys.argv)[3]
        os.mkdir(outDir)
        tmpDir = outDir + "/tmpDir/"
        os.mkdir(tmpDir)

        # read config file
        config = SafeConfigParser()
        config.read(configFile)
        nProcs = int(config.get("main", "nProcs"))
        pool = Pool(processes = nProcs)

        runROBOCOP_EM(coordFile, config, outDir, tmpDir, pool)
