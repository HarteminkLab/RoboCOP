import os
import sys
import robocop
from robocop.utils.readWriteOps import *
from robocop.utils.robocopExtras import *
import robocop.utils.parameterize as parameterize
import numpy as np
import pickle
import re
from configparser import ConfigParser
import robocop.utils.getReads as getReads
import robocop.utils.readData as readData
import pandas
import gc
import h5py
import random
random.seed(9)

# create posterior table for each segment
def createInstances(tf_prob, dbf_conc, coords, pwm, cshared, tmpDir, info_file, fasta_file, nucleosome_file, nucleotide_sequence, mnaseParams, tech):
    segments = len(coords)
    dshared = {} 
    dshared["robocopC"] = cshared 
    robocop.createSharedDictionary(dshared, fasta_file, nucleosome_file, tf_prob, dbf_conc['background'], dbf_conc['nucleosome'], pwm, tmpDir, info_file, nucleotide_sequence)
    for t in range(segments):
        createInstance((t, dshared, coords.iloc[t]['chr'], coords.iloc[t]['start'], coords.iloc[t]['end']))
    if mnaseParams != None:
        for s in range(segments):
            updateMNaseEMMatNB((s, dshared, mnaseParams, tech))
    for t in range(segments):
        posterior_forward_backward_wrapper((t, dshared))
    gc.collect()
    return dshared
    
def runROBOCOP_EM(coordFile, config, outDir, tmpDir, info_file_name, mnaseFile, dnaseFiles = ""):

    info_file = h5py.File(info_file_name, mode = 'w') 
    fragRangeLong = tuple([int(x) for x in re.findall(r'\d+', config.get("main", "fragRangeLong"))])
    fragRangeShort = tuple([int(x) for x in re.findall(r'\d+', config.get("main", "fragRangeShort"))])
    fragRange = (fragRangeLong, fragRangeShort)
    nucFile = config.get("main", "nucFile")
    mnaseFiles = mnaseFile 
    cshared = config.get("main", "cshared")
    tech = config.get("main", "tech")
    # chromosome segments in pandas data frame
    coords = pandas.read_csv(coordFile, sep = "\t")

    # select suubset for training 
    if len(coords) > 500: 
        idx = list(range(len(coords)))
        random.shuffle(idx)
        idx = idx[:500]
        coords = coords.iloc[idx]
        coords = coords.reset_index()
        
    # dbf weights initlaized using KD values
    dbf_conc, pwm = parameterize.getDBFconc(nucFile, config.get("main", "pwmFile"), outDir)

    # read nucleotide sequence and return 1 if successful
    nucleotide_sequence = getReads.getNucSequence(nucFile, tmpDir, info_file, coords)
    # read MNase-seq midpoint counts for long and short fragments

    # readData.get2DValues(mnaseFiles, config.get('main', 'chrSizesFile'), (0, 200), tmpDir)
    # mnase_data_long, mnase_data_short = getReads.getMNaseSmoothed(tmpDir, coords, fragRange, tech = tech)

    mnase_data_long, mnase_data_short = getReads.getMNase(mnaseFiles, tmpDir, info_file, coords, fragRange, tech = tech)
    
    # make t copies of each tf_prob for every timepoint -- only 1 timepoint
    timepoints = len(coords)
    segments = len(coords)
    tf_prob = dict()
    threshold = 0
    thresholds = []
    tName = []
    # determine max prob for TFs to perform constrained EM optimization
    for i in list(dbf_conc.keys()):
        if i != 'nucleosome' and i != 'background':
            tf_prob[i] = dbf_conc[i] 
            if threshold < tf_prob[i] and i != 'unknown':
                threshold = tf_prob[i]
            if i != 'unknown': thresholds.append(tf_prob[i])
            tName.append((i, tf_prob[i]))
    threshold = np.mean(thresholds) + 2*np.std(thresholds)
    # get MNase-seq count parameters
    if mnaseFiles: 
        # with open(outDir + "/negParamsMNase.pkl", 'rb') as readFile:
        #     mnaseParams = pickle.load(readFile, encoding = 'latin1')
        mnaseParams = parameterize.getParamsMNase(mnaseFiles, config.get("main", "nucleosomeFile"), config.get("main", "tfFile"), fragRange, tmpDir, tech)
    else:
        mnaseParams = None
    
    # create shared dictionary for all segments and build HMM transition matrix
    dshared = createInstances(tf_prob, dbf_conc, coords, pwm, cshared, tmpDir, info_file, config.get("main", "nucFile"), config.get("main", "nucleosomeFile"), nucleotide_sequence, mnaseParams, tech)

    fLike = open(outDir + '/likelihood.txt', 'w')
    likelihood = getLogLikelihood(segments, dshared)
    fLike.write(str(likelihood) + '\n')
    fLike.close()
    iterations = 10
    countMNase = 0

    print("Writing MNase params")
    if mnaseFiles != "":
        with open(outDir + "/negParamsMNase.pkl", 'wb') as writeFile:
            pickle.dump(mnaseParams, writeFile, pickle.HIGHEST_PROTOCOL)

    for i in range(iterations):

        #create new dshared
        dsharedNew = {}
        for k in list(dshared.keys()):
            if k == 'info_file': dsharedNew['info_file_name'] = info_file_name
            else: dsharedNew[k] = dshared[k]
        # print("Writing to HMMconfig")
        with open(outDir + "/HMMconfig" + str(i) + ".pkl", 'wb') as writeFile:
            pickle.dump(dsharedNew, writeFile, pickle.HIGHEST_PROTOCOL)

        
        # Baum-Welch on transition probabilities
        background_prob, _tf_prob, nucleosome_prob = update_transition_probs(dshared, segments, tmpDir, threshold)
        tf_prob = np.array([_tf_prob[_] for _ in np.array(sorted(_tf_prob.keys()), order = 'c')])
        robocop.set_transition(dshared, tf_prob, background_prob, nucleosome_prob)
        robocop.set_initial_probs(dshared)

        # posterior decoding with updated transition probabilities
        for t in range(segments):
            setValuesPosterior((t, dshared, tf_prob, background_prob, nucleosome_prob, tmpDir))

        likelihood = getLogLikelihood(segments, dshared)
        fLike = open(outDir + '/likelihood.txt', 'a')
        fLike.write(str(likelihood) + '\n')
        fLike.close()
    
    #create new dshared
    dsharedNew = {}
    for k in list(dshared.keys()):
        if k == 'info_file': dsharedNew['info_file_name'] = info_file_name
        else: dsharedNew[k] = dshared[k]
    print("Writing to HMMconfig")
    with open(outDir + "/HMMconfig.pkl", 'wb') as writeFile:
        pickle.dump(dsharedNew, writeFile, pickle.HIGHEST_PROTOCOL)

    # remove tmpDir
    if len(coords) <= 500: os.system ("rm -rf " + outDir + "tmpDir")

    
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python robocop_em.py <coordinate file> <config file> <output directory>")
        exit(1)
    coordFile = sys.argv[1]
    configFile = sys.argv[2]
    outDir = sys.argv[3]

    run_robocop_with_em(coordFile, configFile, outDir)
