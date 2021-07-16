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
def createInstances(dshared, mnaseParams, coords, cshared, tmpDir, nucleotide_sequence, tech, pool):
    segments = len(coords)
    dshared['tmpDir'] = tmpDir
    dshared['nucleotides'] = nucleotide_sequence
    dshared['robocopC'] = cshared    
    pool.map(createInstance, [(t, dshared) for t in range(segments)])
    if mnaseParams != None: list(map(updateMNaseEMMatNB, [(s, dshared, mnaseParams, tech) for s in range(segments)]))
    pool.map(posterior_forward_backward_wrapper, [(t, dshared) for t in range(segments)])
    gc.collect()
    return dshared
    
def runROBOCOP_NO_EM(coordFile, config, outDir, tmpDir, trainOutDir, pool, dnaseFiles = ""):

    fragRangeLong = tuple([int(x) for x in re.findall(r'\d+', config.get("main", "fragRangeLong"))])
    fragRangeShort = tuple([int(x) for x in re.findall(r'\d+', config.get("main", "fragRangeShort"))])
    fragRange = (fragRangeLong, fragRangeShort)
    nucFile = config.get("main", "nucFile")
    mnaseFiles = config.get("main", "bamFile")
    cshared = config.get("main", "cshared")

    # chromosome segments in data frame
    coords = pandas.read_csv(coordFile, sep = "\t")

    # get info from learned model
    cfg = pickle.load(open(trainOutDir + "/HMMconfig.pkl", "rb"), encoding = 'latin1')
    j = 0
    dbf_conc = {}
    for i in sorted(cfg['tfs']):
        if i == 'background' or i == 'nucleosome':
            continue
        dbf_conc[i] = cfg['tf_prob'][j]
        j += 1
    dbf_conc['background'] = cfg['background_prob']
    dbf_conc['nucleosome'] = cfg['nucleosome_prob']

    # read nucleotide sequence and return 1 if successful
    nucleotide_sequence = getReads.getNucSequence(nucFile, tmpDir, coords)
    # read MNase-seq midpoint counts of long and short fragments
    mnase_data_long, mnase_data_short = getReads.getMNase(mnaseFiles, tmpDir, coords, fragRange)

    # make t copies of each tf_prob for every timepoint -- 1 timepoint
    timepoints = len(coords)
    segments = len(coords)
    tf_prob = dict()
    for i in list(dbf_conc.keys()):
        if i != 'nucleosome' and i != 'background':
            tf_prob[i] = dbf_conc[i]

    # get MNase-seq midpoint count parameters
    if mnaseFiles: 
        with open(trainOutDir + "/negParamsMNase.pkl", 'rb') as readFile:
            mnaseParams = pickle.load(readFile, encoding = 'latin1')
    else:
        mnaseParams = None
    
    dshared = cfg
    tech = config.get("main", "tech")
    createInstances(dshared, mnaseParams, coords, cshared, tmpDir, nucleotide_sequence, tech, pool)
    
    fLike = open(outDir + '/likelihood.txt', 'w')
    print("Likelihood before EM:", getLogLikelihood(segments, dshared['tmpDir']))
    print("Exiting before EM")

    likelihood = getLogLikelihood(segments, dshared['tmpDir'])
    print("Likelihood after EM:", likelihood) 
    return likelihood

if __name__ == '__main__':
        if len(sys.argv) != 5:
                print("Usage: python robocop_no_em.py <coordinate file> <config file> <trained output dir> <outputDir>")
                exit(1)
        coordFile = sys.argv[1]
        configFile = sys.argv[2]
        trainOutDir = (sys.argv)[3]
        outDir = (sys.argv)[4]
        os.makedirs(outDir, exist_ok = True)
        tmpDir = outDir + "/tmpDir/"
        os.makedirs(tmpDir, exist_ok = True)

        # read config file
        config = SafeConfigParser()
        config.read(configFile)
        nProcs = int(config.get("main", "nProcs"))
        pool = Pool(processes = nProcs)

        runROBOCOP_NO_EM(coordFile, config, outDir, tmpDir, trainOutDir, pool)
