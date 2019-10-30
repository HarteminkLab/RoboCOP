from robocop import robocop
from robocop.utils import concentration_probability_conversion, getNucleotides, readData
import numpy as np

# read nucleotide sequence for given segments
def getNucSequence(nucFile, tmpDir, coords):
    nucleotide_sequence = 1
    if nucFile:
        for i, r in coords.iterrows():
            np.save(tmpDir + "nucleotides.idx" + str(i), np.array(getNucleotides.getNucleotideSequence(nucFile, r['chr'], r['start'], r['end'])))
    else: nucleotide_sequence = None
    return nucleotide_sequence

# read MNase midpoint counts (both long and short fragments) for given segments
def getMNase(mnaseFile, tmpDir, coords, fragRange):
    mnase_data_long = ""
    mnase_data_short = ""
    if mnaseFile != "":
        for i, r in coords.iterrows():
            np.save(tmpDir + "MNaseLong.idx" + str(i), np.array(readData.getValuesMNaseOneFileFragmentRange(mnaseFile, r['chr'], r['start'], r['end'], fragRange[0])))
            np.save(tmpDir + "MNaseShort.idx" + str(i), np.array(readData.getValuesMNaseOneFileFragmentRange(mnaseFile, r['chr'], r['start'], r['end'], fragRange[1])))
    return mnase_data_long, mnase_data_short
