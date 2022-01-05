"""
Convert concentration to transition probabilities that 
sum to 1, so that the output of ROBOCOP stays the same. Or convert probabilities to concentration

ROBOCOP needs to set initial probabilities to 0 for states other than
the start of the motif. And only allow end at the last position of motif. 
Otherwise, the results will be slightly different.
"""
import numpy as np
import sys

def solve_for_unbound(motif_lens, motif_conc):
    
    max_len = max(motif_lens)

    p = [0] * (max_len+1)

    p[max_len] = -1

    for i in range(len(motif_lens)):
        motif_len = motif_lens[i]
        p[max_len - motif_len] += motif_conc[i]

    unbound_prob = np.roots(p)
    unbound_prob = unbound_prob[np.isreal(unbound_prob)]
    unbound_prob = unbound_prob[unbound_prob.real > 0]
    unbound_prob = unbound_prob[unbound_prob.real < 1.0]

    if len(unbound_prob) != 1:
        sys.exit('Found %d solutions!' % len(unbound_prob))
        
    else:
        unbound_prob = unbound_prob.real[0]

    return unbound_prob

def convert_to_prob(dbf_conc, pwm):
    """
    To convert concentration to probabilities, we need to solve the following:
    \sum_i motifConcentration_i * p^{motifLen_i} - 1 = 0
    where p is the unboud/background transition probability. 
    """
    motif_len = []
    motif_conc = []

    dbfs = list(dbf_conc.keys())

    for dbf in dbfs:
        if dbf == 'background':
            motif_len.append(1)
            motif_conc.append(dbf_conc['background'])
        elif dbf == 'nucleosome':
            motif_len.append(147)
            motif_conc.append(dbf_conc['nucleosome'])
        elif dbf in pwm:
            motif_len.append(pwm[dbf].shape[1])
            motif_conc.append(dbf_conc[dbf])
        else:
            sys.exit('error: %s not found' % dbf)

    unbound_prob = solve_for_unbound(motif_len, motif_conc)
    
    # calculate the other dbf's prob
    motif_len = np.array(motif_len)
    motif_conc = np.array(motif_conc)

    motif_prob = motif_conc * (unbound_prob ** motif_len)
    assert motif_prob.sum() > 1.0 - 1e-5

    return dict(list(zip(dbfs, motif_prob)))

def convert_to_conc(dbf_prob, pwm):
    """
    the dbf_conc = dbf_prob / (background_prob ^ dbf_len)
    """

    dbf_conc = dict(dbf_prob)
    for dbf in dbf_prob:
        if dbf == 'background':
            dbf_len = 1
        elif dbf == 'nucleosome':
            dbf_len = 147
        elif dbf in pwm:
            dbf_len = pwm[dbf].shape[1]
        else:
            sys.exit('error: %s not found' % dbf)

        dbf_conc[dbf] = dbf_prob[dbf] / (dbf_prob['background'] ** dbf_len)

    return dbf_conc
