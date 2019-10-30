
import os

###############################################################################
## Specify the file that contains the dinucleotide model parameters
## Ideally, that should be placed within this file, but it's separated for now
###############################################################################
nuc_dinucleotide_model_file = os.path.dirname(os.path.realpath(__file__)) +\
								'/nuc_dinucleotide_model.txt'
# the nuc model length is 531 if using 4 states per position
nuc_model_length = 531 
