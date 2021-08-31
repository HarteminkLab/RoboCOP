#####################################################
# This script extract nucleotides from a FASTA file
# into a numpy array. Note that the start and stop
# positions are one indexed and the stop position
# is inclusive.
#####################################################

from Bio import SeqIO
import numpy as np

# Convert nucleotides to 0, 1, 2, 3
def mapNucToInt(n):
    if n == 'A' or n == 'a':
        return 0
    elif n == 'C' or n == 'c':
        return 1
    elif n == 'G' or n == 'g':
        return 2
    elif n == 'T' or n == 't':
        return 3
    else: return 4 # N nucleotide

# extract nucleotide sequence for given chromosome, start and stop                                                                                                                                          
def getNucleotideSequence(fastaFile, chromosome, start = 0, stop = 0):
    fastaSeq = list(SeqIO.parse(open(fastaFile), 'fasta'))
    fastaSequence = {}
    sequenceLengths = {}
    for fs in fastaSeq:
        fastaSequence[fs.name] = fs.seq
        sequenceLengths[fs.name] = len(fs.seq)
    if stop > sequenceLengths[chromosome]:
        print("ERROR: Invalid stop position for chromosome", chromosome, start, stop, sequenceLengths[chromosome])
        exit(1)
    if start <= 0:
        print("ERROR: Invalid start position for chromosome", chromosome)
        exit(1)

    sequence = fastaSequence[chromosome][(start - 1) : stop]
    sequence = np.asarray([mapNucToInt(x) for x in sequence])
    return sequence
