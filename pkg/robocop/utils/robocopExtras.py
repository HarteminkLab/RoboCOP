import robocop
from .readWriteOps import *

# save posterior distribution of DBFs in csv format
def printPosterior(segments, dshared, tmpDir, outDir):
    for s in range(segments):
        x = loadIdx(tmpDir, s)
        robocop.print_posterior_binding_probability(x, dshared, file_name = outDir + '/em_test.out')

# adjust TF weights so that it does not exceed a threshold
def adjustEM(dbf_posterior_start_probs_same_update, dshared, threshold):
    indices = list(range(len(dbf_posterior_start_probs_same_update)))
    thresholdSum = 0
    for i in range(1, dshared['n_tfs']):
        # set to threshold when exceeds threshold
        if dbf_posterior_start_probs_same_update[i] > threshold:
            dbf_posterior_start_probs_same_update[i] = threshold
            thresholdSum += threshold
            indices.remove(i)
    # normalize the rest
    dbf_posterior_start_probs_same_update[indices] = dbf_posterior_start_probs_same_update[indices] / np.sum(dbf_posterior_start_probs_same_update[indices])
    dbf_posterior_start_probs_same_update[indices] *= 1 - thresholdSum
    return dbf_posterior_start_probs_same_update

# Baum Welch update of transition probabilities
def update_transition_probs(dshared, segments, tmpDir, threshold):
    nucleosome_prob = 0 
    background_prob = 0 
    tf_starts = dshared['tf_starts'] 
    tf_lens = dshared['tf_lens'] 
    tfs = dshared['tfs']
    p_table = np.zeros((dshared['n_obs'], dshared['n_states']))
    for t in range(segments):
        x = loadIdx(tmpDir, t)
        p_table += x['posterior_table']
    dbf_posterior_start_probs_same_update = np.empty(dshared['n_tfs'] + 1 + 1) # assuming nucs always present
    dbf_posterior_start_probs_same_update[0] = p_table[:,0].sum()
    for i in range(dshared['n_tfs']):
        dbf_posterior_start_probs_same_update[i + 1] = np.sum(p_table[:,tf_starts[i]])
        dbf_posterior_start_probs_same_update[i + 1] += np.sum(p_table[:,tf_starts[i] + tf_lens[i]])
    dbf_posterior_start_probs_same_update[dshared['n_tfs'] + 1] = np.sum(p_table[:, dshared['nuc_start']])

    # re normalize
    dbf_posterior_start_probs_same_update_em = dbf_posterior_start_probs_same_update / np.sum(dbf_posterior_start_probs_same_update)
    
    # active constraint based EM to limit the max probability for any TF excepting unknown
    # unknown is the last TF in the list
    if np.any(dbf_posterior_start_probs_same_update_em[1 : dshared['n_tfs']] > threshold):
        dbf_posterior_start_probs_same_update_em = adjustEM(dbf_posterior_start_probs_same_update_em, dshared, threshold)
    background_prob = dbf_posterior_start_probs_same_update_em[0] 
    tf_prob = dict()
    for i in range(dshared['n_tfs']):
        tf_prob[tfs[i]] = dbf_posterior_start_probs_same_update_em[i+1] 
    # if self.nuc_present: # assuming always present
    nucleosome_prob = dbf_posterior_start_probs_same_update_em[dshared['n_tfs'] + 1]
    return background_prob, tf_prob, nucleosome_prob

# update data emission matrix using negative binomial distribution parameters
def updateMNaseEMMatNB(args):
    (t, dshared, countParams, tech) = args
    x = loadIdx(dshared['tmpDir'], t)
    robocop.update_data_emission_matrix_using_mnase_midpoint_counts_onePhi(x, dshared, nuc_phi = countParams['nucLong']['phi'], nuc_mus = countParams['nucLong']['mu']*countParams['nucLong']['scale'], tf_phi = countParams['tfLong']['phi'], tf_mu = countParams['tfLong']['mu'], other_phi = countParams['otherLong']['phi'], other_mu = countParams['otherLong']['mu'], mnaseType = 'long', tech = tech)
    robocop.update_data_emission_matrix_using_mnase_midpoint_counts_onePhi(x, dshared, nuc_phi = countParams['nucShort']['phi'], nuc_mus = countParams['nucShort']['mu']*countParams['nucShort']['scale'], tf_phi = countParams['tfShort']['phi'], tf_mu = countParams['tfShort']['mu'], other_phi = countParams['otherShort']['phi'], other_mu = countParams['otherShort']['mu'], mnaseType = 'short', tech = tech)
    dumpIdx(x, dshared['tmpDir'])

# Posterior decoding
def setValuesPosterior(args):
    (t, dshared, tf_prob, background_prob, nucleosome_prob, tmpDir) = args
    x = loadIdx(tmpDir, t)
    robocop.posterior_forward_backward(x, dshared)
    dumpIdx(x, dshared['tmpDir'])

# Compute log likelihood
def getLogLikelihood(segments, tmpDir):
    logLikelihood = 0
    for s in range(segments):
        x = loadIdx(tmpDir, s)
        logLikelihood += robocop.get_log_likelihood(x)
    return logLikelihood

def build_data_emission_matrix_wrapper(t):
    x = loadIdx(tmpDir, t)
    robocop.build_data_emission_matrix(x)
    dumpIdx(x, tmpDir)

# wrapper function to perform posterior decoding
def posterior_forward_backward_wrapper(args):
    (t, dshared) = args
    x = loadIdx(dshared['tmpDir'], t)
    robocop.posterior_forward_backward(x, dshared)
    dumpIdx(x, dshared['tmpDir'])

def viterbi_decoding_wrapper(args):
    (t, dshared) = args
    x = loadIdx(dshared['tmpDir'], t)
    robocop.viterbi_decoding(x, dshared)
    np.save(dshared['tmpDir'] + "viterbi_traceback.idx" + str(x['segment']), x['viterbi_traceback'])
    np.save(dshared['tmpDir'] + "viterbi_traceback.idx" + str(x['segment']), x['viterbi_table'])
    
# create dictionary for each segment
def createInstance(args):
    (t, dshared) = args
    x = robocop.createDictionary(t, dshared)
    dumpIdx(x, dshared['tmpDir'])
    
