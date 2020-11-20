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
    # manager = Manager()
    dshared = {} #manager.dict()
    dshared["robocopC"] = cshared 
    robocop.createSharedDictionary(dshared, tf_prob, dbf_conc['background'], dbf_conc['nucleosome'], pwm, tmpDir, nucleotide_sequence)
    for t in range(segments):
        createInstance((t, dshared))
    # map(createInstance, [(t, dshared) for t in range(segments)])
    if mnaseParams != None:
        for s in range(segments):
            updateMNaseEMMatNB((s, dshared, mnaseParams, tech))
        # list(map(updateMNaseEMMatNB, [(s, dshared, mnaseParams) for s in range(segments)]))
    for t in range(segments):
        posterior_forward_backward_wrapper((t, dshared))
    # map(posterior_forward_backward_wrapper, [(t, dshared) for t in range(segments)])
    gc.collect()
    return dshared
    
def runROBOCOP_EM(coordFile, config, outDir, tmpDir, pool, mnaseFile, dnaseFiles = ""):

    fragRangeLong = tuple([int(x) for x in re.findall(r'\d+', config.get("main", "fragRangeLong"))])
    fragRangeShort = tuple([int(x) for x in re.findall(r'\d+', config.get("main", "fragRangeShort"))])
    fragRange = (fragRangeLong, fragRangeShort)
    nucFile = config.get("main", "nucFile")
    mnaseFiles = mnaseFile #config.get("main", "mnaseFile")
    cshared = config.get("main", "cshared")
    tech = config.get("main", "tech")
    # chromosome segments in pandas data frame
    coords = pandas.read_csv(coordFile, sep = "\t")

    # dbf weights initlaized using KD values
    dbf_conc, pwm = parameterize.getDBFconc(nucFile, config.get("main", "pwmFile"))

    # read nucleotide sequence and return 1 if successful
    nucleotide_sequence = getReads.getNucSequence(nucFile, tmpDir, coords)
    # read MNase-seq midpoint counts for long and short fragments
    mnase_data_long, mnase_data_short = getReads.getMNase(mnaseFiles, tmpDir, coords, fragRange, tech = tech)
    
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
    # threshold = 10000
    print("Threshold is:", threshold)
    # exit(0)
    # get MNase-seq count parameters
    if mnaseFiles: 
        mnaseParams = parameterize.getParamsMNase(mnaseFiles, config.get("main", "nucleosomeFile"), config.get("main", "tfFile"), fragRange, tech)
    else:
        mnaseParams = None
    
    # create shared dictionary for all segments and build HMM transition matrix
    dshared = createInstances(tf_prob, dbf_conc, coords, pwm, cshared, tmpDir, nucleotide_sequence, mnaseParams, tech, pool)

    # # Delete nucleotide and MNase files
    # os.system("rm dshared['tmpDir'] + nucleotides*npy")
    # os.system("rm dshared['tmpDir'] + MNase*npy")
    
    fLike = open(outDir + '/likelihood.txt', 'w')
    likelihood = getLogLikelihood(segments, dshared['tmpDir'])
    fLike.write(str(likelihood) + '\n')
    fLike.close()
    # print("Likelihood before EM:", getLogLikelihood(segments, dshared['tmpDir']))
    iterations = 10 #150 #200 #40
    countMNase = 0
    lfMNaseLong = 1
    lfMNaseShort = 1
    lfMNaseScale = 1

    print("Writing MNase params")
    if mnaseFiles != "":
        with open(outDir + "/negParamsMNase.pkl", 'wb') as writeFile:
            pickle.dump(mnaseParams, writeFile, pickle.HIGHEST_PROTOCOL)

    for i in range(iterations):
        # Baum-Welch on transition probabilities
        background_prob, _tf_prob, nucleosome_prob = update_transition_probs(dshared, segments, tmpDir, threshold)
        tf_prob = np.array([_tf_prob[_] for _ in np.array(sorted(_tf_prob.keys()), order = 'c')])
        robocop.set_transition(dshared, tf_prob, background_prob, nucleosome_prob)
        robocop.set_initial_probs(dshared)

        # posterior decoding with updated transition probabilities
        for t in range(segments):
            setValuesPosterior((t, dshared, tf_prob, background_prob, nucleosome_prob, tmpDir))

    #     dsharedNew = {}
    #     for k in list(dshared.keys()):
    #         dsharedNew[k] = dshared[k]
    #     print("Writing to HMMconfig")
    #     with open(outDir + "/HMMconfig" + str(i) + ".pkl", 'wb') as writeFile:
    #         pickle.dump(dsharedNew, writeFile, pickle.HIGHEST_PROTOCOL)


    #     likelihood = getLogLikelihood(segments, dshared['tmpDir'])
    #     # print("Likelihood in iter:", i, getLogLikelihood(segments, dshared['tmpDir']))
    #     fLike = open(outDir + '/likelihood.txt', 'a')
    #     fLike.write(str(likelihood) + '\n')
    #     fLike.close()
    # # printPosterior(segments, dshared, tmpDir, outDir)
    
    likelihood = getLogLikelihood(segments, dshared['tmpDir'])
    fLike = open(outDir + '/likelihood.txt', 'a')
    # print("Likelihood after EM:", likelihood) #x.get_log_likelihood()
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
    os.system ("rm -rf " + outDir + "tmpDir")

if __name__ == '__main__':
        if len(sys.argv) != 4:
                print("Usage: python robocop_em.py <coordinate file> <config file> <output directory>")
                exit(1)

        # times = ["DMAH_64_82", "DMAH_65_83", "DMAH_66_84", "DMAH_67_85", "DMAH_68_86", "DMAH_69_87", "DMAH_71_88", "DMAH_72_89", "DMAH_73_90", "DMAH_74_91", "DMAH_75_92", "DMAH_76_93", "DMAH_77_94", "DMAH_78_95", "DMAH_79_96"] # , "DMAH_80_97", "DMAH_81_98"]
        # filenames = ["/usr/xtmp/sneha/data/MNase-seq/MacAlpine_cell_cyle/" + x + "/" + x + "_sacCer3_subsampled.bam" for x in times]
        # outDirs = ["/usr/xtmp/sneha/RoboCOP_cell_cycle_new/" + x + "_subsampled_Chr4/" for x in times]
        # filenames = ["/usr/xtmp/sneha/data/MNase-seq/MacAlpine_cell_cyle/" + x + "/" + x + "_sacCer3.bam" for x in times]
        # outDirs = ["/usr/xtmp/sneha/RoboCOP_cell_cycle_new/" + x + "_allMotifs_Chr4/" for x in times]
        # outDirs = ["/usr/xtmp/sneha/RoboCOP_cell_cycle_new/" + x + "_shortFrag_100_nucFrag_136_196_Chr4/" for x in times]
        # outDirs = ["/usr/xtmp/sneha/RoboCOP_cell_cycle_new/" + x + "_shortFrag_100_nucFrag_131_191_Chr4/" for x in times]
        # outDirs = ["/usr/xtmp/sneha/RoboCOP_cell_cycle_new/" + x + "_shortFrag_120_nucFrag_136_196_Chr4/" for x in times]
        coordFile = sys.argv[1]
        configFile = sys.argv[2]
        outDir = sys.argv[3]
    
        print("RoboCOP: model training ...")
        print("Coordinates: " + coordFile)
        print("Config file: " + configFile)
        print("Output dir: " + outDir)

        # outDir = (sys.argv)[3]
        # counter = int(sys.argv[3])

        # outDir = "/usr/xtmp/sneha/robocop_os60_Chr4_sf0_80_nf127_187/"
        # outDir = outDirs[counter]
        # outDir = "/usr/xtmp/sneha/robocop_DM504_sacCer3/"
        # outDir = "/usr/xtmp/sneha/robocop_DM504_sacCer3_test/"
        # bamFile = "/usr/xtmp/sneha/data/MNase-seq/MacAlpine/DM504/dm504.bam"
        # bamFile = "/usr/project/compbio/tqtran/cd/DM504/DM504_sacCer3_m1_2020-05-20-18-48.bam"
        # bamFile = filenames[counter]

        # outDir = "/usr/xtmp/sneha/robocop_DM508_new/"
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
