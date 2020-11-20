from robocop import robocop
from robocop.utils import concentration_probability_conversion, getNucleotides, readData
import numpy as np

# read nucleotide sequence for given segments
def getNucSequence(nucFile, tmpDir, coords, idx = None):
    nucleotide_sequence = 1
    if idx != None:
        np.save(tmpDir + "nucleotides.idx" + str(idx), np.array(getNucleotides.getNucleotideSequence(nucFile, coords.iloc[idx]['chr'], coords.iloc[idx]['start'], coords.iloc[idx]['end'])))
        return nucleotide_sequence
        
    if nucFile:
        for i, r in coords.iterrows():
            # print("Coords:", r['chr'], r['start'], r['end'])
            np.save(tmpDir + "nucleotides.idx" + str(i), np.array(getNucleotides.getNucleotideSequence(nucFile, r['chr'], r['start'], r['end'])))
    else: nucleotide_sequence = None
    return nucleotide_sequence

# read MNase midpoint counts (both long and short fragments) for given segments
def getMNase(mnaseFile, tmpDir, coords, fragRange, idx = None, tech = "MNase"):
    mnase_data_long = ""
    mnase_data_short = ""

    # offset bu 4 if ATAC-seq otherwise no offset
    offset = 4 if tech == "ATAC" else 0
    if mnaseFile != "":
        if idx != None:
            longs = readData.getValuesMNaseOneFileFragmentRange(mnaseFile, coords.iloc[idx]['chr'], coords.iloc[idx]['start'], coords.iloc[idx]['end'], fragRange[0], offset)
            shorts = readData.getValuesMNaseOneFileFragmentRange(mnaseFile, coords.iloc[idx]['chr'], coords.iloc[idx]['start'], coords.iloc[idx]['end'], fragRange[1], offset)
            np.save(tmpDir + tech + "Long.idx" + str(idx), np.array(longs))
            np.save(tmpDir + tech + "Short.idx" + str(idx), np.array(shorts))
            return mnase_data_long, mnase_data_short

        for i, r in coords.iterrows():
            np.save(tmpDir + tech + "Long.idx" + str(i), np.array(readData.getValuesMNaseOneFileFragmentRange(mnaseFile, r['chr'], r['start'], r['end'], fragRange[0], offset)))
            np.save(tmpDir + tech + "Short.idx" + str(i), np.array(readData.getValuesMNaseOneFileFragmentRange(mnaseFile, r['chr'], r['start'], r['end'], fragRange[1], offset)))
    return mnase_data_long, mnase_data_short
