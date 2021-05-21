###########################################################
# Get all motifs from motif file in MEME format
############################################################

import numpy as np
import pickle
import pandas
import sys

def convertMotifFormat(motifFile, outputFile):
    motifLines = open(motifFile, "r").readlines()
    motifLines = [x for x in motifLines if x != "\n"]
    
    foundMotif = 0
    motifDict = {}

    for i in motifLines:
        l = i.strip()
        if l[:5] == "MOTIF":
            l = l.split()
            foundMotif = l[1]
            motifDict[foundMotif] = [[], [], [], []]
            continue
        if foundMotif != 0 and l[:6] != "letter":
            l = l.split()
            for j in range(4):
                motifDict[foundMotif][j] = motifDict[foundMotif][j] + [float(l[j])]

    for m in list(motifDict.keys()):
        motifDict[m] = np.array(motifDict[m])

    for m in list(motifDict.keys()):
        motifDict[m] = {'matrix': motifDict[m]}

    # add unknown factor
    motifDict['unknown'] = {'source': "Motif of background sequence", "matrix": np.array([[0.308512, 0.308512, 0.308512, 0.308512, 0.308512, 0.308512, 0.308512, 0.308512, 0.308512, 0.308512], [0.191488, 0.191488, 0.191488, 0.191488, 0.191488, 0.191488, 0.191488, 0.191488, 0.191488, 0.191488], [0.191488, 0.191488, 0.191488, 0.191488, 0.191488, 0.191488, 0.191488, 0.191488, 0.191488, 0.191488], [0.308512, 0.308512, 0.308512, 0.308512, 0.308512, 0.308512, 0.308512, 0.308512, 0.308512, 0.308512]])}

    # add background 
    motifDict["background"] = {'source':"# A,C,G,T frequency in yeast genome, from Todd's ruby code", 'matrix': np.array([[0.308512], [0.191488], [0.191488], [0.308512]])}

    # save file
    pickle.dump(motifDict, open(outputFile, "wb"))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("USAGE: python getMotifsFromMEME.py <motif file in MEME format> <output pickle file>")
        exit(1)
    motifFile = (sys.argv)[1]
    outputFile = (sys.argv)[2]
    if outputFile[-2:] != ".p":
        print("ERROR: Output file must be a pickle file with .p extension. Example: output.p")
        exit(1)
    convertMotifFormat(motifFile, outputFile)
