import os
import sys
sys.path.append("/home/home3/sneha/.local/lib/python3.6/site-packages/")
sys.path.append(os.getcwd())
import robocop
from robocop.utils.readWriteOps import *
from robocop.utils.robocopExtras import *
import robocop.utils.parameterize as parameterize
import numpy as np
import pickle
import re
from configparser import SafeConfigParser
import robocop.utils.getReads as getReads
import pandas
import gc
from multiprocessing import Pool, Manager

# create posterior table for each segment
def createInstances(tf_prob, dbf_conc, coords, pwm, cshared, tmpDir, nucleotide_sequence, mnaseParams, tech, pool):
    segments = len(coords)
    dshared = {} 
    dshared["robocopC"] = cshared 
    robocop.createSharedDictionary(dshared, tf_prob, dbf_conc['background'], dbf_conc['nucleosome'], pwm, tmpDir, nucleotide_sequence)
    for t in range(segments):
        createInstance((t, dshared))
    if mnaseParams != None:
        for s in range(segments):
            updateMNaseEMMatNB((s, dshared, mnaseParams, tech))
    for t in range(segments):
        posterior_forward_backward_wrapper((t, dshared))
    gc.collect()
    return dshared
    
def runROBOCOP_EM(coordFile, config, outDir, tmpDir, pool, mnaseFile, dnaseFiles = ""):

    fragRangeLong = tuple([int(x) for x in re.findall(r'\d+', config.get("main", "fragRangeLong"))])
    fragRangeShort = tuple([int(x) for x in re.findall(r'\d+', config.get("main", "fragRangeShort"))])
    fragRange = (fragRangeLong, fragRangeShort)
    nucFile = config.get("main", "nucFile")
    mnaseFiles = mnaseFile 
    cshared = config.get("main", "cshared")
    tech = config.get("main", "tech")
    # chromosome segments in pandas data frame
    coords = pandas.read_csv(coordFile, sep = "\t")

    # dbf weights initlaized using KD values
    dbf_conc, pwm = parameterize.getDBFconc(nucFile, config.get("main", "pwmFile"))

    # read nucleotide sequence and return 1 if successful
    nucleotide_sequence = getReads.getNucSequence(nucFile, tmpDir, coords)

    # # read MNase-seq midpoint counts for long and short fragments
    # mnase_data_long, mnase_data_short = getReads.getMNase(mnaseFiles, tmpDir, coords, fragRange, tech = tech)
    
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
                threshold = tf_prob[i][0]
            if i != 'unknown': thresholds.append(tf_prob[i][0])
            tName.append((i, tf_prob[i][0]))
    print(thresholds)
    print(tName)
    threshold = np.mean(thresholds) + 2*np.std(thresholds)
    # # get MNase-seq count parameters
    # if mnaseFiles: 
    #     mnaseParams = parameterize.getParamsMNase(mnaseFiles, config.get("main", "nucleosomeFile"), config.get("main", "tfFile"), fragRange, tech)
    # else:
    #     mnaseParams = None

    mnaseParams = None
    
    # create shared dictionary for all segments and build HMM transition matrix
    dshared = createInstances(tf_prob, dbf_conc, coords, pwm, cshared, tmpDir, nucleotide_sequence, mnaseParams, tech, pool)

    fLike = open(outDir + '/likelihood.txt', 'w')
    likelihood = getLogLikelihood(segments, dshared['tmpDir'])
    fLike.write(str(likelihood) + '\n')
    fLike.close()
    iterations = 0 # 10
    countMNase = 0
    lfMNaseLong = 1
    lfMNaseShort = 1
    lfMNaseScale = 1

    # print("Writing MNase params")
    # if mnaseFiles != "":
    #     with open(outDir + "/negParamsMNase.pkl", 'wb') as writeFile:
    #         pickle.dump(mnaseParams, writeFile, pickle.HIGHEST_PROTOCOL)

    for i in range(iterations):
        # Baum-Welch on transition probabilities
        background_prob, _tf_prob, nucleosome_prob = update_transition_probs(dshared, segments, tmpDir, threshold)
        tf_prob = np.array([_tf_prob[_] for _ in np.array(sorted(_tf_prob.keys()), order = 'c')])
        robocop.set_transition(dshared, tf_prob, background_prob, nucleosome_prob)
        robocop.set_initial_probs(dshared)

        # posterior decoding with updated transition probabilities
        for t in range(segments):
            setValuesPosterior((t, dshared, tf_prob, background_prob, nucleosome_prob, tmpDir))

    likelihood = getLogLikelihood(segments, dshared['tmpDir'])
    fLike = open(outDir + '/likelihood.txt', 'a')
    fLike.write(str(likelihood) + '\n')
    fLike.close()
    
    #create new dshared
    dsharedNew = {}
    for k in list(dshared.keys()):
        dsharedNew[k] = dshared[k]
    print("Writing to HMMconfig")
    with open(outDir + "/HMMconfig.pkl", 'wb') as writeFile:
        pickle.dump(dsharedNew, writeFile, pickle.HIGHEST_PROTOCOL)

    # remove tmpDir
    # os.system ("rm -rf " + outDir + "tmpDir")

if __name__ == '__main__':
        if len(sys.argv) != 4:
                print("Usage: python robocop_em.py <coordinate file> <config file> <output directory>")
                exit(1)
        coordFile = sys.argv[1]
        configFile = sys.argv[2]
        outDir = sys.argv[3]
    
        print("RoboCOP: model training ...")
        print("Coordinates: " + coordFile)
        print("Config file: " + configFile)
        print("Output dir: " + outDir)

        os.makedirs(outDir, exist_ok = True)
        tmpDir = outDir + "/tmpDir/"
        os.makedirs(tmpDir, exist_ok = True)
        
        # copy config file
        os.system("cp " + configFile + " " + outDir + "/config.ini") 
        # copy coordinates file
        os.system("cp " + coordFile + " " + outDir + "/coords.tsv") 

        configFile = outDir + "/config.ini"

        fc = open(configFile, 'a')
        try:
            fc.write("\ntrainDir = " + outDir + "\n")
            fc.close()
        except configparser.DuplicateOptionError:
            fc.close()

        # read config file
        config = SafeConfigParser()
        config.read(configFile)
        nProcs = int(config.get("main", "nProcs"))
        # pool = Pool(processes = nProcs)
        if not config.has_option("main", "bamFile"): bamFile = None
        else: bamFile = config.get("main", "bamFile")

        runROBOCOP_EM(coordFile, config, outDir, tmpDir, 1, bamFile)
