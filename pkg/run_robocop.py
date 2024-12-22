##############################################################################################
# USAGE
## run_robocop_with_em: coordinate_file, config_file, output_directory
## run_robocop_without_em: coordinate_file, trained_output_directory, output_directory,
##                         index (optional for parallel execution),
##                         total_splits (optional for parallel execution)
##############################################################################################    

import os
import pandas
import configparser
from robocop_em import runROBOCOP_EM
from robocop_no_em import runROBOCOP_NO_EM
from robocop.utils import plotRoboCOP

def run_robocop_with_em(coordFile, configFile, outDir):
    if outDir[-1] != '/': outDir += '/'
    
    print("RoboCOP: model training ...")
    print("Coordinates: " + coordFile)
    print("Config file: " + configFile)
    print("Output dir: " + outDir)
    
    os.makedirs(outDir, exist_ok = True)
    tmpDir = outDir + "tmpDir/"
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
    config = configparser.ConfigParser()
    config.read(configFile)
    if not config.has_option("main", "bamFile"): bamFile = None
    else: bamFile = config.get("main", "bamFile")
    
    info_file_name = tmpDir + 'info.h5' 
    runROBOCOP_EM(coordFile, config, outDir, tmpDir, info_file_name, bamFile)

def run_robocop_without_em(coordFile, trainOutDir, outDir, idx = None, total = None):
    if outDir[-1] != '/': outDir += '/'
    configFile = trainOutDir + "/config.ini"
    
    os.makedirs(outDir, exist_ok = True)
    tmpDir = outDir + "/tmpDir/"
    os.makedirs(tmpDir, exist_ok = True)
    
    # copy config file
    os.system("cp " + configFile + " " + outDir + "/config.ini") 
    # copy coordinates file
    os.system("cp " + coordFile + " " + outDir + "/coords.tsv") 
    
    # copy nucleosome dinucleotide model
    os.system("cp " + trainOutDir + "/nuc_dinucleotide_model.txt " + outDir + "/")
    os.system("cp " + trainOutDir + "/nuc_emission.npy " + outDir + "/")
    
    # copy pwm file
    os.system("cp " + trainOutDir + "/pwm.p " + outDir + "/")
    
    configFile = outDir + "config.ini"
    
    print("RoboCOP: model fitting ...")
    print("Coordinates: " + coordFile)
    print("Config file: " + configFile)
    print("Train dir: " + trainOutDir)
    print("Output dir: " + outDir)
    
    
    # chromosome segments in data frame
    coords = pandas.read_csv(coordFile, sep = "\t")
    coords = coords.reset_index()

    # idx = int((sys.argv)[4])
    config = configparser.ConfigParser()
    read_ok = config.read(configFile)
    print("Config sections:", config.sections(), configFile)
    if not read_ok: 
        print("Read failed",  idx)
        exit(0)
    bamFile = config.get("main", "bamFile")

    if idx is None:
        info_file_name = tmpDir + 'info.h5'
    else:
        info_file_name = tmpDir + 'info_' + str(idx) + '_' + str(total) + '.h5'
        indices = [i for i in range(idx, len(coords), total)]
        coords = coords.iloc[indices]
        
    runROBOCOP_NO_EM(coords, config, outDir, tmpDir, info_file_name, trainOutDir, bamFile)


def plot_robocop_output(robocop_out_dir, chrm, start, end, save = True):
    plotRoboCOP.plot_output(robocop_out_dir, chrm, start, end, save)
