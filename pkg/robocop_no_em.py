import robocop
from robocop.utils.readWriteOps import *
from robocop.utils.robocopExtras import *
import robocop.utils.parameterize as parameterize
import numpy as np
import pickle
import os
import re
import sys
import configparser
import robocop.utils.getReads as getReads
import pandas
import gc

# create posterior table for each segment
def createInstances(dshared, mnaseParams, coords, cshared, tmpDir, nucleotide_sequence, segment, tech):

    dshared['tmpDir'] = tmpDir
    dshared['nucleotides'] = nucleotide_sequence
    dshared['robocopC'] = cshared
    
    createInstance((segment, dshared, coords['chr'], coords['start'], coords['end']))
    if mnaseParams != None: updateMNaseEMMatNB((segment, dshared, mnaseParams, tech))
    posterior_forward_backward_wrapper((segment, dshared))
    gc.collect()
    return dshared
    
def runROBOCOP_NO_EM(coords, config, outDir, tmpDir, info_file_name, trainOutDir, mnaseFiles, dnaseFiles = ""):

    info_file = h5py.File(info_file_name, mode = 'w') 

    fragRangeLong = tuple([int(x) for x in re.findall(r'\d+', config.get("main", "fragRangeLong"))])
    fragRangeShort = tuple([int(x) for x in re.findall(r'\d+', config.get("main", "fragRangeShort"))])
    fragRange = (fragRangeLong, fragRangeShort)
    nucFile = config.get("main", "nucFile")
    tech = config.get("main", "tech")
    cshared = config.get("main", "cshared")

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

    '''
    #######################
    nuc_prob = 1e-30
    print("Old BG:", cfg['background_prob'])
    sums = 0
    for i in dbf_conc:
        if i == 'nucleosome': continue
        sums += dbf_conc[i]
    dbf_conc['nucleosome'] = nuc_prob
    cfg['nucleosome_prob'] = nuc_prob
    cfg['background_prob'] /= sums * (1 - nuc_prob)
    for i in range(cfg['n_tfs']):
        cfg['tf_prob'][i] /= sums * (1 - nuc_prob)
    for i in dbf_conc:
        dbf_conc[i] = dbf_conc[i] / sums * (1 - nuc_prob)

    _, pwm = parameterize.getDBFconc(nucFile, config.get("main", "pwmFile"), outDir)
    robocop.robocop.build_transition_matrix(cfg, pwm, nuc_dinucleotide_model_file = cfg['tmpDir'] + '/../nuc_dinucleotide_model.txt', tf_prob = cfg['tf_prob'], background_prob = cfg['background_prob'], nucleosome_prob = cfg['nucleosome_prob'], allow_end_at_any_state = 1)
    print(dbf_conc)
    print(cfg)
    print(cfg['initial_probs'])
    print(cfg['tf_prob'])
    print(cfg['initial_probs'].shape, cfg['tf_prob'].shape)
    exit(0)
    #######################
    '''
    
    # read nucleotide sequence and return 1 if successful
    nucleotide_sequence = getReads.getNucSequence(nucFile, tmpDir, info_file, coords)
    # read MNase-seq midpoint counts of long and short fragments
    mnase_data_long, mnase_data_short = getReads.getMNase(mnaseFiles, tmpDir, info_file, coords, fragRange, tech = tech)

    # make t copies of each tf_prob for every timepoint -- 1 timepoint
    timepoints = 1
    segments = 1
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
    dshared['info_file'] = info_file

    for idx, r in coords.iterrows():
        createInstances(dshared, mnaseParams, r, cshared, tmpDir, nucleotide_sequence, idx, tech)

    
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python robocop_no_em.py <coordinate file> <trained output directory> <output directory> <index -- optional> <total splits -- optional>")
        exit(1)
    coordFile = sys.argv[1]
    trainOutDir = (sys.argv)[2]
    outDir = (sys.argv)[3]
        
    # split coords to run in parallel
    if len(sys.argv) == 6:
        idx = int((sys.argv)[4])
        total = int((sys.argv)[5])

