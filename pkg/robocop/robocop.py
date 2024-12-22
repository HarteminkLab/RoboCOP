from ctypes import *
import numpy as np
from numpy.ctypeslib import ndpointer
from scipy.stats import nbinom, gamma
import sys, os
from .nucleosome.calc_dinucleotide import getDiNuc
import pickle
from Bio import SeqIO
import math
import pandas
from scipy import sparse

def update_sparse_posterior(f, k, v):
    v = np.array(v)
    v[v < 1e-4] = 0
    v_sparse = sparse.csr_matrix(v)
    g = f[k+'/data'] # f.create_group(k)
    g[...] = v_sparse.data
    g = f[k+'/indices'] # f.create_group(k)
    g[...] = v_sparse.indices
    g = f[k+'/indptr'] # f.create_group(k)
    g[...] = v_sparse.indptr
    # g.create_dataset('data', data=v_sparse.data)
    # g.create_dataset('indices', data=v_sparse.indices)
    # g.create_dataset('indptr', data=v_sparse.indptr)
    g.attrs['shape'] = v_sparse.shape

def save_sparse_posterior(f, k, v):
    v = np.array(v)
    v[v < 1e-4] = 0
    v_sparse = sparse.csr_matrix(v)
    g = f.create_group(k)
    g.create_dataset('data', data=v_sparse.data)
    g.create_dataset('indices', data=v_sparse.indices)
    g.create_dataset('indptr', data=v_sparse.indptr)
    g.attrs['shape'] = v_sparse.shape

def get_sparse(f, k):
    g = f[k]
    v_sparse = sparse.csr_matrix((g['data'][:],g['indices'][:], g['indptr'][:]), g.attrs['shape'])
    return v_sparse

def get_sparse_todense(f, k):
    v_dense = np.array(get_sparse(f, k).todense())
    if v_dense.shape[0]==1: v_dense = v_dense[0]
    return v_dense

def reverse_complement(pwm):
    """
    Given a pwm (4 by n numpy array), return the reverse complement pwm
    """
    rc_pwm = np.zeros((5, len(pwm[0,])))
    rc_pwm[0, ] = pwm[3, ::-1]
    rc_pwm[1, ] = pwm[2, ::-1]
    rc_pwm[2, ] = pwm[1, ::-1]
    rc_pwm[3, ] = pwm[0, ::-1]
    return rc_pwm

def createDictionary(segment, dshared, chrm, start, end):
    """
    create dictionary for one segment and construct and save
    data emission matrix for the segment.
    """
    d = {}
    d['segment'] = segment
    d['chr'] = chrm
    d['start'] = start
    d['end'] = end
    d['n_obs'] = end - start + 1
    build_data_emission_matrix(d, dshared, segment, d['n_obs'])
    d['log_likelihood'] = None
    d['posterior_table'] = None
    return d

def createSharedDictionary(d, fasta_file, nucleosome_file, tf_prob, background_prob, nucleosome_prob, pwm, tmpDir, info_file, nt_array = None, n_obs = None, lock = None):
    """
    create shared dictionary with shared information.
    Create HMM transition matrix.
    """
    tfs = np.array(list(tf_prob.keys()))
    tfs.sort()
    d['n_tfs'] = len(tfs)
    _tf_prob = np.array([tf_prob[_] for _ in tfs])
    d['timepoints'] = 1 # 1 timepoint only
    d['padding'] = 0 # TF motif padding is 0
    d['tmpDir'] = tmpDir
    d['background_prob'] = background_prob
    d['nucleosome_prob'] = nucleosome_prob
    d['tf_prob'] = _tf_prob
    d['tfs'] = tfs
    d['nucleotides'] = nt_array
    d['info_file'] = info_file
    check_parameters(d)
    # create nucleosome dinucleotide model
    if not os.path.isfile(d['tmpDir'] + '../nuc_dinucleotide_model.txt') and not os.path.isfile(d['tmpDir'] + '../nuc_emission.npy'):
        nuc_emission = getDiNuc(nucleosome_file, fasta_file, d['tmpDir'] + '/../nuc_dinucleotide_model.txt')
        np.save(d['tmpDir'] + '/../nuc_emission', nuc_emission)

    else:
        nuc_emission = np.load(d['tmpDir'] + '/../nuc_emission.npy')
    # build the HMM transition matrix
    build_transition_matrix(d, pwm, nuc_dinucleotide_model_file = d['tmpDir'] + '/../nuc_dinucleotide_model.txt', tf_prob = _tf_prob, background_prob = background_prob, nucleosome_prob = nucleosome_prob, allow_end_at_any_state = 1)
    if nt_array != None: stack_pwms(d, pwm, nuc_emission)

#### transition matrix related code 
def check_parameters(d):
    background_prob = d['background_prob']
    nucleosome_prob = d['nucleosome_prob']
    tf_prob = d['tf_prob']
    tfs = d['tfs']

    if d['n_tfs'] == 0 and nucleosome_prob == 0:
        sys.exit('Must have at least of one TF or nucleosome')
    for t in range(d['timepoints']):
        for i in range(d['n_tfs']):
            prob = tf_prob[i]
            if prob < 0:
                sys.exit('%s prob %f is less than 0' % (tfs[i, t], prob))
        if background_prob <= 0:
            sys.exit('must have background and background prob must be > 0')
        if nucleosome_prob < 0:
            sys.exit('nucleosome prob must be >= 0')
    # the background conc should be 1 
    # otherwise the sum of the prob should be 1
    for t in range(d['timepoints']):
        if background_prob != 1:
            prob_sum = np.sum(tf_prob[:]) + background_prob +\
                       nucleosome_prob
            if abs(prob_sum - 1) > 1e-9:
                error_message = 'The sum of all prob is %.20f' % prob_sum
                error_message += ' but it should should be 1'
                error_message += ', or the background conc should be set to 1.'
                sys.exit(error_message)

def build_transition_matrix(d, pwm, nuc_dinucleotide_model_file, tf_prob, background_prob = 0,
                            nucleosome_prob = 0, allow_end_at_any_state = 1):
    """Build the backbone of the transition matrix. Set transition inside
    motifs to be 1. Set the nucleosome dinucleotide mode if nuc is present."""
    get_transition_matrix_info(d, pwm, allow_end_at_any_state)
    _build_transition_matrix(d, nuc_dinucleotide_model_file)
    set_transition(d, [], background_prob, nucleosome_prob)
    set_initial_probs(d)
    set_end_probs(d)
        
def get_transition_matrix_info(d, pwm, allow_end_at_any_state):
    """
    Obtain the information needed to construct the transition matrix,
    based on DBFs in the model and PWM used. 
    """
    nucleosome_prob = d['nucleosome_prob']
    nuc_model_length = 531
    tfs = d['tfs']
    # adding padding to TFs on both sides
    if d['nucleotides'] is None: tf_lens = np.array([(10 + 2*d['padding']) for tf in tfs])
    else: tf_lens = np.array(
        [(pwm[tf].shape[1] + 2*d['padding']) for tf in tfs]) # setting default TF length to 10
    # total number of states 
    if nucleosome_prob > 0:
        n_states = 1 + 2*np.sum(tf_lens) + nuc_model_length
        nuc_present = 1
    else:
        n_states = 1 + 2*np.sum(tf_lens)
        nuc_present = 0
    # plus the silent states, one for each tf and the central silent state
    n_states += d['n_tfs'] + 1
    # the start state of each TF 
    # first tf always starts from state 1, unless no tf is in this model
    # tf_starts includes padding
    tf_starts = np.array([1] * d['n_tfs'])
    for i in range(1, d['n_tfs']):
        tf_starts[i] = tf_starts[i - 1] + 2 * tf_lens[i - 1]
    # the start state of each nucleosome
    if nuc_present:
        if d['n_tfs'] > 0:
            nuc_start = tf_starts[d['n_tfs'] - 1] +\
                        2 * tf_lens[d['n_tfs'] - 1]
        else:
            nuc_start = 1
        nuc_len = nuc_model_length
    else:
        nuc_start = 0
        nuc_len = 0
    # the start of silent state
    if nuc_present:
        silent_states_begin = nuc_start + nuc_model_length
    else:
        if d['n_tfs'] > 0:
            silent_states_begin = \
                                  tf_starts[d['n_tfs'] - 1] + 2 * tf_lens[d['n_tfs'] - 1]
        else:
            silent_states_begin = 1
    ## allow hidden state sequence to end at any state or not
    d['end_at_any_state'] = allow_end_at_any_state
    ## information to preserve for later use
    d['nuc_present'] = nuc_present
    d['nuc_start'] = nuc_start
    d['nuc_len'] = nuc_len 
    d['n_states'] = n_states
    # hard coding for now
    d['n_vars'] = 5
    d['silent_states_begin'] = silent_states_begin
    d['tf_starts'] = tf_starts
    d['tf_lens'] = tf_lens

# construct transition matrix
def _build_transition_matrix(d, nuc_dinucleotide_model_file):
    robocopC = CDLL(d["robocopC"])
    tf_starts = d['tf_starts']
    tf_lens = d['tf_lens']
    # make sure the nucleosome dinucleotide model file exists 
    robocopC.construct_transition_matrix.argtypes = [c_int, c_int, c_int, ndpointer(np.long, flags = "C_CONTIGUOUS"), ndpointer(np.long, flags = "C_CONTIGUOUS"), c_int, c_int, ndpointer(np.double, flags = "C_CONTIGUOUS"), POINTER(c_char)]
    if d['nuc_present']:
        if not os.path.isfile(nuc_dinucleotide_model_file):
            raise IOError(
                '%s does not exits. Make sure the package installation is correct.'
                % nuc_dinucleotide_model_file)
    for t in range(d['timepoints']):
        t_mat = np.zeros((d['n_states'], d['n_states'])) 
        if d['n_tfs'] > 0:
            tf_starts = tf_starts.astype(np.long)
            tf_lens = tf_lens.astype(np.long)
            t_mat = t_mat.astype(np.double)
            robocopC.construct_transition_matrix(
                d['n_states'], d['nuc_present'], d['nuc_start'], 
                tf_starts, tf_lens, 
                d['n_tfs'], d['silent_states_begin'], t_mat,
                c_char_p(nuc_dinucleotide_model_file.encode('utf-8')))
        else:
            robocopC.construct_transition_matrix(
                d['n_states'], d['nuc_present'], d['nuc_start'], 
                NULL, NULL, 
                d['n_tfs'], d['silent_states_begin'], t_mat, 
                c_char_p(nuc_dinucleotide_model_file.encode('utf-8')))
        d['transition_matrix'] = t_mat

def stack_pwms(d, pwm, nuc_emission):
    # background must be in the model
    tfs = d['tfs']
    pwm_emat = np.transpose(pwm['background'])
    for tf in tfs:
        # add padding -- padding is 0 so no padding
        for i in range(d['padding']): pwm_emat = np.vstack((pwm_emat, np.transpose(pwm['background'])))
        pwm_emat = np.vstack((pwm_emat, np.transpose(pwm[tf])))
        # add padding -- padding is 0 so no padding
        for i in range(d['padding']): pwm_emat = np.vstack((pwm_emat, np.transpose(pwm['background'])))
        
        # add padding -- padding is 0 so no padding
        for i in range(d['padding']): pwm_emat = np.vstack((pwm_emat, np.transpose(pwm['background'])))
        pwm_emat = np.vstack((pwm_emat, np.transpose(
            reverse_complement(pwm[tf]))))
        # add padding -- padding is 0 so no padding
        for i in range(d['padding']): pwm_emat = np.vstack((pwm_emat, np.transpose(pwm['background'])))


    if d['nuc_present']:
        pwm_emat = np.vstack((pwm_emat, nuc_emission))
    assert pwm_emat.shape[0] == d['silent_states_begin']
    _pwm_emat = np.ascontiguousarray(pwm_emat)
    d['pwm_emission'] = _pwm_emat
    
def _build_data_emission_matrix(d, dshared, segment, n_obs):
    # 3d matrix -- dim 0 is for timepoints, 
    # dim 1 denotes the multivariate emission values
    # right now it's (0) nucleotide, (1) mnase-short, (2) mnase-long, (3) atac-short (4) atac-long
    # need to remove hard codedness in future
    info_file = dshared['info_file']
    if n_obs is None: n_obs = info_file['segment_' + str(segment)].attrs['n_obs']
    for t in range(dshared['timepoints']):
        data_emat = np.ones((5, n_obs, dshared['n_states']))
        if dshared['nucleotides'] is not None: update_data_emission_matrix_using_nucleotides(data_emat, dshared, segment, n_obs)
    d['emission'] = data_emat
    # k = 'segment_' + str(segment) + '/emission'
    # if k not in info_file.keys():
    #     g_emat = info_file.create_dataset(k, data = data_emat)
    # else:
    #     g_emat = info_file[k]
    #     g_emat[...] = data_emat
        
def build_data_emission_matrix(d, dshared, segment, n_obs = None):
    _build_data_emission_matrix(d, dshared, segment, n_obs)

def update_data_emission_matrix_using_nucleotides(data_emission_matrix, dshared, segment, n_obs):
    robocopC = CDLL(dshared["robocopC"])
    robocopC.build_emission_mat_from_pwm.argtypes = [ndpointer(np.long, flags = "C_CONTIGUOUS"), ndpointer(np.double, flags = "C_CONTIGUOUS"), c_int, c_int, c_int, c_int, ndpointer(np.double, flags = "C_CONTIGUOUS")]
    # data_emission_matrix = d['data_emission_matrix']
    info_file = dshared['info_file']
    for t in range(dshared['timepoints']):
        nucleotides = get_sparse_todense(info_file, 'segment_'+str(segment)+'/nucleotides')
        # nucleotides = info_file['segment_' + str(segment) + '/nucleotides'][:] # np.load(dshared['tmpDir'] + "nucleotides.idx" + str(segment) + ".npy")
        data_emat = data_emission_matrix[0]
        pwm = dshared['pwm_emission']
        nucleotides = nucleotides.astype(np.long)
        pwm = pwm.astype(np.double)
        data_emat = data_emat.astype(np.double)

        robocopC.build_emission_mat_from_pwm(nucleotides, pwm, n_obs, dshared['n_states'], dshared['silent_states_begin'], 5, data_emat)
        data_emission_matrix[0] = data_emat



def update_data_emission_matrix_using_mnase_midpoint_counts_norm(
            d, dshared, nuc_mean, nuc_sd, tf_mean, tf_sd, other_mean, other_sd, mnaseType):
    tf_starts = dshared['tf_starts']
    tf_lens = dshared['tf_lens']
    mnaseData = np.load(dshared['tmpDir'] + "kernelized_counts_" + mnaseType + "_" + d['chr'] + ".npy")[d['start'] - 1 : d['end']]
    means = np.zeros(dshared['silent_states_begin'])
    sds = np.zeros(dshared['silent_states_begin'])
    means[:] = other_mean
    sds[:] = other_sd
    for i in range(dshared['n_tfs']):
        tf_start = tf_starts[i]
        tf_end = tf_start + 2 * tf_lens[i]
        means[tf_start:tf_end] = tf_mean
        sds[tf_start:tf_end] = tf_sd
    if dshared['nuc_present']:
        means[dshared['nuc_start']:(dshared['nuc_start']+9)] = nuc_mean[0:9]
        sds[dshared['nuc_start']:(dshared['nuc_start']+9)] = nuc_sd[0:9]
        for i in range(128):
            means[(dshared['nuc_start']+9 + i*4):((dshared['nuc_start'] + 9 + i*4 + 4))] = nuc_mean[i+9]
            sds[(dshared['nuc_start']+9 + i*4):((dshared['nuc_start'] + 9 + i*4 + 4))] = nuc_sd[i+9]
        means[(dshared['nuc_start'] + 9 + 128*4):(dshared['nuc_start'] + 9 + 128*4 + 10)] = nuc_mean[137:147]
        sds[(dshared['nuc_start'] + 9 + 128*4):(dshared['nuc_start'] + 9 + 128*4 + 10)] = nuc_sd[137:147]
    assert (means == 0).sum() == 0
    assert (sds == 0).sum() == 0

    # update data emission matrix
    dictionary = {}
    if mnaseType == "tf": m_idx = 1
    else: m_idx = 2
    for i in range(d['data_emission_matrix'].shape[1]):
        for j in range(dshared['silent_states_begin']):
            val = 1/(sds[j]*math.sqrt(2*math.pi))*math.exp(-pow(mnaseData[i] - means[j], 2)/(2*pow(sds[j], 2)))
            d['data_emission_matrix'][m_idx][i][j] *= val



def update_data_emission_matrix_using_mnase_midpoint_counts_gamma(
            d, dshared, nuc_shape, nuc_rate, tf_shape, tf_rate, other_shape, other_rate, mnaseType):
    tf_starts = dshared['tf_starts']
    tf_lens = dshared['tf_lens']
    mnaseData = np.load(dshared['tmpDir'] + "kernelized_counts_" + mnaseType + "_" + d['chr'] + ".npy")[d['start'] - 1 : d['end']]
    shape = np.zeros(dshared['silent_states_begin'])
    rate = np.zeros(dshared['silent_states_begin'])
    shape[:] = other_shape
    rate[:] = other_rate
    for i in range(dshared['n_tfs']):
        tf_start = tf_starts[i]
        tf_end = tf_start + 2 * tf_lens[i]
        shape[tf_start:tf_end] = tf_shape
        rate[tf_start:tf_end] = tf_rate
    if dshared['nuc_present']:
        shape[dshared['nuc_start']:(dshared['nuc_start']+9)] = nuc_shape[0:9]
        rate[dshared['nuc_start']:(dshared['nuc_start']+9)] = nuc_rate[0:9]
        for i in range(128):
            shape[(dshared['nuc_start']+9 + i*4):((dshared['nuc_start'] + 9 + i*4 + 4))] = nuc_shape[i+9]
            rate[(dshared['nuc_start']+9 + i*4):((dshared['nuc_start'] + 9 + i*4 + 4))] = nuc_rate[i+9]
        shape[(dshared['nuc_start'] + 9 + 128*4):(dshared['nuc_start'] + 9 + 128*4 + 10)] = nuc_shape[137:147]
        rate[(dshared['nuc_start'] + 9 + 128*4):(dshared['nuc_start'] + 9 + 128*4 + 10)] = nuc_rate[137:147]
    assert (shape == 0).sum() == 0
    assert (rate == 0).sum() == 0

    # update data emission matrix
    dictionary = {}
    if mnaseType == "tf": m_idx = 1
    else: m_idx = 2
    for i in range(d['data_emission_matrix'].shape[1]):
        for j in range(dshared['silent_states_begin']):
            val = gamma.pdf(mnaseData[i], a = shape[j], scale = 1/rate[j])
            d['data_emission_matrix'][m_idx][i][j] *= val


def update_data_emission_matrix_using_negative_binomial(
            d, segment, dshared, phis, mus, data, index, timepoint):
    """
    Update the data emission matrix based on the negative binomial
    distribution.
    This function allows using different phi and mu for every single state.
    phis: an array containing phi for every non-silent state 
    mus:  an array containing mu for every non-silent state 
    data: an array of integer data_emission_matrix
    """
    info_file = dshared['info_file']
    # data_emission_matrix = info_file['segment_' + str(segment) + '/emission'][:] # d['data_emission_matrix']
    data_emission_matrix = d['emission']
    n_obs = info_file['segment_' + str(segment)].attrs['n_obs']
    dictionary = {}
    for i in range(n_obs):
        for j in range(dshared['silent_states_begin']):
            if (phis[j], mus[j]) not in dictionary:
                dictionary[(phis[j], mus[j])] = {}
                p = phis[j]/(mus[j] + phis[j])
                dictionary[(phis[j], mus[j])][data[i]] = nbinom.pmf(data[i], phis[j], p)
            elif data[i] not in dictionary[(phis[j], mus[j])]:
                p = phis[j]/(mus[j] + phis[j])
                dictionary[(phis[j], mus[j])][data[i]] = nbinom.pmf(data[i], phis[j], p)
            data_emission_matrix[index][i][j] *= dictionary[(phis[j], mus[j])][data[i]]
    # emat = info_file['segment_' + str(segment) + '/emission']
    # emat[...] = data_emission_matrix
    d['emission'] = data_emission_matrix
    
def update_data_emission_matrix_using_mnase_midpoint_counts_onePhi(
            d, segment, dshared, nuc_phi, nuc_mus, tf_phi, tf_mu, other_phi, other_mu, mnaseType, tech = "MNase"):
    """
    Update data emission matrix using N.B.
    """
    tf_starts = dshared['tf_starts']
    tf_lens = dshared['tf_lens']
    k = 'segment_' + str(segment) + '/' 
    info_file = dshared['info_file']
    if mnaseType == 'short':
        mnaseData = get_sparse_todense(info_file, k+tech+'_short')
        # mnaseData = info_file[k + tech + '_short'][:]
    else:
        mnaseData = get_sparse_todense(info_file, k+tech+'_long')
        # mnaseData = info_file[k + tech + '_long'][:]
    phis = np.zeros(dshared['silent_states_begin'])
    mus = np.zeros(dshared['silent_states_begin'])
    phis[:] = other_phi
    mus[:] = other_mu
    for i in range(dshared['n_tfs']):
        tf_start = tf_starts[i]
        tf_end = tf_start + 2 * tf_lens[i]
        phis[tf_start:tf_end] = tf_phi
        mus[tf_start:tf_end] = tf_mu
    if dshared['nuc_present']:
        mus[dshared['nuc_start']:(dshared['nuc_start']+9)] = nuc_mus[0:9]
        for i in range(128):
            mus[(dshared['nuc_start']+9 + i*4):((dshared['nuc_start'] + 9 + i*4 + 4))] = nuc_mus[i+9]
        mus[(dshared['nuc_start'] + 9 + 128*4):(dshared['nuc_start'] + 9 + 128*4 + 10)] = nuc_mus[137:147]
        phis[dshared['nuc_start']:(dshared['nuc_start'] + dshared['nuc_len'])] = nuc_phi
    assert (mus == 0).sum() == 0
    assert (phis == 0).sum() == 0
    for t in range(dshared['timepoints']):
        if mnaseType == 'short' and tech == "MNase": update_data_emission_matrix_using_negative_binomial(d, segment, dshared, phis, mus, mnaseData, 1, t)
        elif mnaseType == 'short' and tech == "ATAC": update_data_emission_matrix_using_negative_binomial(d, segment, dshared, phis, mus, mnaseData, 3, t)
        elif mnaseType == 'long' and tech == "MNase": update_data_emission_matrix_using_negative_binomial(d, segment, dshared, phis, mus, mnaseData, 2, t)
        elif mnaseType == 'long' and tech == "ATAC": update_data_emission_matrix_using_negative_binomial(d, segment, dshared, phis, mus, mnaseData, 4, t)

def set_transition(d, tf_prob, background_prob, nucleosome_prob):
    """
    transitions is a dict of transition prob/weights for the dbfs in the model.
    e.g. {'Abf1':0.01, 'Cbf1':0.01, 'background':0.98}
    transitions that will be modified are the transitions from central
    silent state to the begining silent state of each tf,
    central silent state to background state, as well as central silent
    state to nucleosome start state
    """
    for t in range(d['timepoints']):
        if tf_prob == []: tf_prob = d['tf_prob']
        else:
            d['tf_prob'] = tf_prob
            d['background_prob'] = background_prob
            d['nucleosome_prob'] = nucleosome_prob
            
        t_mat = d['transition_matrix']
        t_mat[d['silent_states_begin'], 0] = background_prob
        if d['nuc_present']:
            t_mat[d['silent_states_begin'], d['nuc_start']] = nucleosome_prob
        for i in range(d['n_tfs']):
            t_mat[d['silent_states_begin'], d['silent_states_begin'] + i + 1] = tf_prob[i]
        d['transition_matrix'] = t_mat

def set_initial_probs(d):
    """
    The transitions of each dbf should be set before initial probs are set
    """
    robocopC = CDLL(d["robocopC"])
    robocopC.set_initial_probs.argtypes = [ndpointer(np.long, flags = "C_CONTIGUOUS"), ndpointer(np.long, flags = "C_CONTIGUOUS"), c_int, c_int, c_int, c_int, c_int, c_int, ndpointer(np.double, flags = "C_CONTIGUOUS"), ndpointer(np.double, flags = "C_CONTIGUOUS")]
    initial_probs = np.zeros(d['n_states'])
    motif_starts = d['tf_starts']
    motif_lens = d['tf_lens']
    t_mat = d['transition_matrix']
    if d['n_tfs'] > 0:
        motif_starts = motif_starts.astype(np.long)
        motif_lens = motif_lens.astype(np.long)
        t_mat = t_mat.astype(np.double)
        initial_probs = initial_probs.astype(np.double)
        robocopC.set_initial_probs(
            motif_starts, motif_lens,
            d['n_tfs'], d['silent_states_begin'], d['n_states'], 
            d['nuc_present'], d['nuc_start'], d['nuc_len'],
            t_mat, initial_probs)
    else:
        robocopC.set_initial_probs(
            None, None,
            d['n_tfs'], d['silent_states_begin'], d['n_states'], 
            d['nuc_present'], d['nuc_start'], d['nuc_len'],
            t_mat, initial_probs)
    # set initial probs
    d['initial_probs'] = initial_probs
        
def set_end_probs(d):
    tf_starts = d['tf_starts']
    tf_lens = d['tf_lens']
    if d['end_at_any_state']:
        end_probs = np.ones(d['n_states'])
        d['end_probs'] = end_probs
    else:
        end_probs = np.zeros(d['n_states'])
        # background state
        end_probs[0] = 1.0
        # each of the motif ends
        for i in range(d['n_tfs']):
            end_probs[motif_starts[i] + motif_lens[i] - 1] = 0.5
            end_probs[motif_starts[i] + 2*motif_lens[i] - 1] = 0.5
        # nuc end
        if d['nuc_present']:
            end_probs[d['nuc_start'] + d['nuc_len'] - 1] = 1.0
        d['end_probs'] = end_probs

def posterior_forward_backward_loop(d, dshared, segment):
    """
    Forward backward and posterior decoding.
    """
    k = 'segment_' + str(segment) + '/'
    info_file = dshared['info_file']
    n_obs = info_file[k].attrs['n_obs']
    robocopC = CDLL(dshared["robocopC"])
    motif_starts = dshared['tf_starts'] 
    motif_lens = dshared['tf_lens'] 
    initial_probs = dshared['initial_probs']
    end_probs = dshared['end_probs']
    transition_mat = dshared['transition_matrix']
    data_emission_mat = d['emission'] # info_file[k + 'emission'][:] # d['data_emission_matrix']
    fscaling_factors = np.zeros(n_obs) # d['n_obs'])
    bscaling_factors = np.zeros(n_obs)
    scaling_factors = np.zeros(n_obs)
    parents_mat = np.zeros((dshared['n_states'], dshared['n_states']), dtype = int)
    n_parents = np.zeros(dshared['n_states'], dtype = int)
    children_mat = np.zeros((dshared['n_states'], dshared['n_states']), dtype = int)
    n_children = np.zeros(dshared['n_states'], dtype = int)
    ftable = np.zeros((n_obs, dshared['n_states']))
    btable = np.zeros((n_obs, dshared['n_states']))
    p_table = np.zeros((n_obs, dshared['n_states']))
    parents_mat = parents_mat.astype(np.long)
    children_mat = children_mat.astype(np.long)
    n_parents = n_parents.astype(np.long)
    n_children = n_children.astype(np.long)
    transition_mat = transition_mat.astype(np.double)
    data_emission_mat = data_emission_mat.astype(np.double)
    initial_probs = initial_probs.astype(np.double)
    end_probs = end_probs.astype(np.double)
    ftable = ftable.astype(np.double)
    btable = btable.astype(np.double)
    fscaling_factors = fscaling_factors.astype(np.double)
    bscaling_factors = bscaling_factors.astype(np.double)
    scaling_factors = scaling_factors.astype(np.longdouble)
    p_table = p_table.astype(np.double)
    motif_starts = motif_starts.astype(np.long)
    motif_lens = motif_lens.astype(np.long)
    
    robocopC.find_parents_and_children.argtypes = [ndpointer(np.long, flags = "C_CONTIGUOUS"), ndpointer(np.long, flags = "C_CONTIGUOUS"), ndpointer(np.long, flags = "C_CONTIGUOUS"), ndpointer(np.long, flags = "C_CONTIGUOUS"), c_int, c_int, ndpointer(np.double, flags = "C_CONTIGUOUS")]
    robocopC.find_parents_and_children(parents_mat, children_mat,
                                         n_parents, n_children, dshared['n_states'],
                                         dshared['silent_states_begin'], transition_mat)

    # Forward algorithm
    robocopC.fward.argtypes = [ndpointer(np.double, flags = "C_CONTIGUOUS"), ndpointer(np.double, flags = "C_CONTIGUOUS"), ndpointer(np.double, flags = "C_CONTIGUOUS"), ndpointer(np.double, flags = "C_CONTIGUOUS"), c_int, c_int, c_int, c_int, ndpointer(np.long, flags = "C_CONTIGUOUS"), ndpointer(np.long, flags = "C_CONTIGUOUS"), ndpointer(np.long, flags = "C_CONTIGUOUS"), ndpointer(np.long, flags = "C_CONTIGUOUS"), c_int, c_int, c_int, c_int, ndpointer(np.double, flags = "C_CONTIGUOUS"), ndpointer(np.double, flags = "C_CONTIGUOUS")]
    robocopC.fward(
        initial_probs,
        transition_mat, data_emission_mat,
        end_probs,
        dshared['n_states'], dshared['silent_states_begin'], n_obs, dshared['n_vars'], 
        parents_mat, n_parents,
        motif_starts, motif_lens, dshared['n_tfs'],
        dshared['nuc_present'], dshared['nuc_start'], dshared['nuc_len'],
        ftable, fscaling_factors
    )

    # Backward algorithm
    robocopC.bward.argtypes = [ndpointer(np.double, flags = "C_CONTIGUOUS"), ndpointer(np.double, flags = "C_CONTIGUOUS"), ndpointer(np.double, flags = "C_CONTIGUOUS"), c_int, c_int, c_int, c_int, ndpointer(np.long, flags = "C_CONTIGUOUS"), ndpointer(np.long, flags = "C_CONTIGUOUS"), ndpointer(np.long, flags = "C_CONTIGUOUS"), ndpointer(np.long, flags = "C_CONTIGUOUS"), c_int, c_int, c_int, c_int, ndpointer(np.double, flags = "C_CONTIGUOUS"), ndpointer(np.double, flags = "C_CONTIGUOUS")]
    robocopC.bward(
        transition_mat, data_emission_mat,
        end_probs,
        dshared['n_states'], dshared['silent_states_begin'], n_obs, dshared['n_vars'],
        children_mat, n_children,
        motif_starts, motif_lens, dshared['n_tfs'],
        dshared['nuc_present'], dshared['nuc_start'], dshared['nuc_len'],
        btable, bscaling_factors)

    robocopC.calc_sr.argtypes = [ndpointer(np.double, flags = "C_CONTIGUOUS"), ndpointer(np.double, flags = "C_CONTIGUOUS"), c_int, ndpointer(np.longdouble, flags = "C_CONTIGUOUS")]
    robocopC.calc_sr.restype = c_double
    log_fscaling_factor_sum_loop = robocopC.calc_sr(
        fscaling_factors, bscaling_factors,
        n_obs, scaling_factors)

    # posterior decoding
    robocopC.posterior_decoding.argtypes = [ndpointer(np.double, flags = "C_CONTIGUOUS"), ndpointer(np.double, flags = "C_CONTIGUOUS"), ndpointer(np.double, flags = "C_CONTIGUOUS"), ndpointer(np.longdouble, flags = "C_CONTIGUOUS"), c_int, c_int, ndpointer(np.double, flags = "C_CONTIGUOUS")]
    robocopC.posterior_decoding(
        ftable, btable, bscaling_factors,
        scaling_factors, dshared['n_states'], n_obs, 
        p_table)

    pos_key = k + 'posterior'
    if pos_key not in info_file.keys():
        # g_pos = info_file.create_dataset(pos_key, data = p_table)
        g_pos = save_sparse_posterior(info_file, k+'posterior', p_table)
    else:
        g_pos = update_sparse_posterior(info_file, k+'posterior', p_table)
        # g_pos = info_file[pos_key]
        # g_pos[...] = p_table
    return log_fscaling_factor_sum_loop

def posterior_forward_backward(d, segment, dshared):
    log_fscaling_factor_sum = 0
    info_file = dshared['info_file']
    for t in range(dshared['timepoints']):
        log_fscaling_factor_sum += posterior_forward_backward_loop(d, dshared, segment)
    info_file['segment_' + str(segment)].attrs['log_likelihood'] = log_fscaling_factor_sum


def center_for_dbf_probs(dshared, segment): #, lock = None):
    """
    the length of nucleosome padding is defined here as 0, for now
    This may need to re implemented in C for speed 
    """
    info_file = dshared['info_file']
    n_obs = info_file['segment_' + str(segment)]['n_obs']
    nuc_padding_length = 0 # 5
    nuc_start = dshared['nuc_start']
    nuc_len = dshared['nuc_len'] 
    # the end position of first nucleosome padding background (its start
    # position is just nuc_start)
    nuc_padding_end1 = nuc_start + nuc_padding_length
    actual_nuc_start = nuc_padding_end1
    # the start position of second nucleosome padding background 
    nuc_padding_start2 = nuc_start + nuc_len - nuc_padding_length
    nuc_padding_end2 = nuc_start + nuc_len
    actual_nuc_end = nuc_padding_start2
    # nucleosome center positions, using the 4 states per nucleosotide model
    # the end of first padding, plus 9 normal background states, 1 branching
    # background state, and then half the nucleosome states
    nuc_center_start = nuc_padding_end1 + 9 + 4 + 4 * 63
    nuc_center_end = nuc_center_start + 4
    tf_starts = dshared['tf_starts']
    tf_lens = dshared['tf_lens']
    for t in range(dshared['timepoints']):
        dbf_binding_probs = np.zeros((n_obs, dshared['n_tfs'] + 1 + 5*dshared['nuc_present']))
        posterior_table = info_file['segment_' + str(segment) + '/posterior'][:] # d['posterior_table']
        for i in range(n_obs):
            # background
            dbf_binding_probs[i, 0] = posterior_table[i, 0]
            
            # add padding
            for j in range(dshared['n_tfs']):
                # motif
                # assuming padding = 0
                dbf_binding_probs[i, j + 1] = posterior_table[i, tf_starts[j] + tf_lens[j]/2]
                # reverse motif
                dbf_binding_probs[i, j + 1] += posterior_table[i, tf_starts[j] + tf_lens[j]/2 + tf_lens[j]]
            if dshared['nuc_present']:
                dbf_binding_probs[i, dshared['n_tfs'] + 1] = \
                                                       posterior_table[i, nuc_start:nuc_padding_end1].sum() + \
                                                       posterior_table[i, nuc_padding_start2:nuc_padding_end2].sum()
                dbf_binding_probs[i, dshared['n_tfs'] + 2] = \
                                                       posterior_table[i, nuc_padding_end1:nuc_padding_start2].sum()
                dbf_binding_probs[i, dshared['n_tfs'] + 3] = \
                                                       posterior_table[i, nuc_center_start:nuc_center_end].sum()
                dbf_binding_probs[i, dshared['n_tfs'] + 4] = \
                                                       posterior_table[i, actual_nuc_start]
                dbf_binding_probs[i, dshared['n_tfs'] + 5] = \
                                                       posterior_table[i, actual_nuc_end]
            if dbf_binding_probs[i, dshared['n_tfs'] + 2] > 1.000000001: print("GREATER: ", dbf_binding_probs[t, i, dshared['n_tfs'] + 2])
        return dbf_binding_probs

def sum_for_dbf_probs(dshared, posterior_table):
    """
    the length of nucleosome padding is defined here as 0
    """

    n_obs = posterior_table.shape[0] 
    nuc_padding_length = 0 
    nuc_start = dshared['nuc_start']
    nuc_len = dshared['nuc_len'] 
    # the end position of first nucleosome padding background (its start
    # position is just nuc_start)
    nuc_padding_end1 = nuc_start + nuc_padding_length
    actual_nuc_start = nuc_padding_end1
    # the start position of second nucleosome padding background 
    nuc_padding_start2 = nuc_start + nuc_len - nuc_padding_length
    nuc_padding_end2 = nuc_start + nuc_len
    actual_nuc_end = nuc_padding_start2
    # nucleosome center positions, using the 4 states per nucleosotide model
    # the end of first padding, plus 9 normal background states, 1 branching
    # background state, and then half the nucleosome states
    nuc_center_start = nuc_padding_end1 + 9 + 4 + 4 * 63
    nuc_center_end = nuc_center_start + 4
    tf_starts = dshared['tf_starts']
    tf_lens = dshared['tf_lens']
    for t in range(dshared['timepoints']):
        dbf_binding_probs = np.zeros((n_obs, dshared['n_tfs'] + 1 + 5*dshared['nuc_present']))
        for i in range(n_obs):
            # background
            dbf_binding_probs[i, 0] = posterior_table[i, 0]            
            # add padding

            for j in range(dshared['n_tfs']):
                # motif
                dbf_binding_probs[i, j + 1] = posterior_table[i, (tf_starts[j] + dshared['padding']):(tf_starts[j] + tf_lens[j] - dshared['padding'])].sum()
                # reverse motif
                dbf_binding_probs[i, j + 1] += posterior_table[i, (tf_starts[j] + tf_lens[j]):(tf_starts[j] + 2*tf_lens[j] - dshared['padding'])].sum()
            if dshared['nuc_present']:
                dbf_binding_probs[i, dshared['n_tfs'] + 1] = \
                                                       posterior_table[i, nuc_start:nuc_padding_end1].sum() + \
                                                       posterior_table[i, nuc_padding_start2:nuc_padding_end2].sum()
                dbf_binding_probs[i, dshared['n_tfs'] + 2] = \
                                                       posterior_table[i, nuc_padding_end1:nuc_padding_start2].sum()
                dbf_binding_probs[i, dshared['n_tfs'] + 3] = \
                                                       posterior_table[i, nuc_center_start:nuc_center_end].sum()
                dbf_binding_probs[i, dshared['n_tfs'] + 4] = \
                                                       posterior_table[i, actual_nuc_start]
                dbf_binding_probs[i, dshared['n_tfs'] + 5] = \
                                                       posterior_table[i, actual_nuc_end]

            if dbf_binding_probs[i, dshared['n_tfs'] + 2] > 1.000000001: print("GREATER: ", dbf_binding_probs[i, dshared['n_tfs'] + 2])
        return dbf_binding_probs


def sum_for_dbf_probs_fwd_rev(d, dshared):
    """
    the length of nucleosome padding is defined here as 0
    """
    
    nuc_padding_length = 0 
    nuc_start = dshared['nuc_start']
    nuc_len = dshared['nuc_len'] 
    # the end position of first nucleosome padding background (its start
    # position is just nuc_start)
    nuc_padding_end1 = nuc_start + nuc_padding_length
    actual_nuc_start = nuc_padding_end1
    # the start position of second nucleosome padding background 
    nuc_padding_start2 = nuc_start + nuc_len - nuc_padding_length
    nuc_padding_end2 = nuc_start + nuc_len
    actual_nuc_end = nuc_padding_start2
    # nucleosome center positions, using the 4 states per nucleosotide model
    # the end of first padding, plus 9 normal background states, 1 branching
    # background state, and then half the nucleosome states
    nuc_center_start = nuc_padding_end1 + 9 + 4 + 4 * 63
    nuc_center_end = nuc_center_start + 4
    tf_starts = dshared['tf_starts']
    tf_lens = dshared['tf_lens']
    for t in range(dshared['timepoints']):
        dbf_binding_probs = np.zeros((d['n_obs'], dshared['n_tfs'] + 1 + 1 + 5*dshared['nuc_present']))
        posterior_table = d['posterior_table']
        for i in range(d['n_obs']):
            # background
            dbf_binding_probs[i, 0] = posterior_table[i, 0]            
            # add padding

            opos = list(dshared['tfs']).index('ORC')
            dbf_binding_probs[i, dshared['n_tfs'] + 1] += posterior_table[i, (tf_starts[opos] + tf_lens[opos]):(tf_starts[opos] + 2*tf_lens[opos] - dshared['padding'])].sum()
            
            for j in range(dshared['n_tfs']):
                # motif
                dbf_binding_probs[i, j + 1] = posterior_table[i, (tf_starts[j] + dshared['padding']):(tf_starts[j] + tf_lens[j] - dshared['padding'])].sum()
            
            if dshared['nuc_present']:
                dbf_binding_probs[i, dshared['n_tfs'] + 1 + 1] = \
                                                       posterior_table[i, nuc_start:nuc_padding_end1].sum() + \
                                                       posterior_table[i, nuc_padding_start2:nuc_padding_end2].sum()
                dbf_binding_probs[i, dshared['n_tfs'] + 1 + 2] = \
                                                       posterior_table[i, nuc_padding_end1:nuc_padding_start2].sum()
                dbf_binding_probs[i, dshared['n_tfs'] + 1 + 3] = \
                                                       posterior_table[i, nuc_center_start:nuc_center_end].sum()
                dbf_binding_probs[i, dshared['n_tfs'] + 1 + 4] = \
                                                       posterior_table[i, actual_nuc_start]
                dbf_binding_probs[i, dshared['n_tfs'] + 1 + 5] = \
                                                       posterior_table[i, actual_nuc_end]
            if dbf_binding_probs[i, dshared['n_tfs'] + 1 + 2] > 1.000000001: print("GREATER: ", dbf_binding_probs[t, i, dshared['n_tfs'] + 1 + 2])
        return dbf_binding_probs

# save posterior probability as csv
def print_posterior_binding_probability(dshared, idx, file_name = '', 
                                        by_dbf = True):
    tfs = dshared['tfs']
    info_file = dshared['info_file']
    n_obs = info_file['segment_' + str(idx)].attrs['n_obs']
    for t in range(dshared['timepoints']):
        posterior_table = info_file['segment_' + str(idx) + '/posterior'][:]
        
        if file_name == '':
            from sys import stdout 
            output = stdout 
        else:
            output = open(file_name + '.timepoint' + str(idx), 'w')
        if not by_dbf:
            for i in range(n_obs):
                output.write(
                    '\t'.join(['%g' % _ for _ in posterior_table[i,]]) + '\n'
                )
        else:
            dbf_binding_probs = sum_for_dbf_probs(dshared, posterior_table)
            header = "background"
            if dshared['n_tfs'] > 0:
                header += "\t%s" % '\t'.join(tfs)
            if dshared['nuc_present']:
                header += "\tnuc_padding\tnucleosome\tnuc_center\tnuc_start\tnuc_end"
            output.write(header + '\n')
            for i in range(n_obs):
                output.write('\t'.join(['%g' % _ for _ in dbf_binding_probs[i, ]]) + '\n')
        if file_name != '':
            output.close()


# save posterior probability as csv
def get_posterior_binding_probability_df(dshared, posterior_table):
    tfs = dshared['tfs']
    info_file = dshared['info_file']
    dbf_binding_probs = sum_for_dbf_probs(dshared, posterior_table)
    header = ["background"]
    header += list(dshared['tfs'])
    if dshared['nuc_present']:
        header += ["nuc_padding", "nucleosome", "nuc_center", "nuc_start", "nuc_end"]
    dbf_binding_probs_df = pandas.DataFrame(dbf_binding_probs, columns = header)
    return dbf_binding_probs_df


############################################
## Functions for extracting internal data ##
############################################

def get_states_info(d):
    return d['n_states'], d['silent_states_begin']

def get_initial_probs(d):
    return d['initial_probs'] 

def get_pwm_emission(d):
    return d['pwm_emission']

def get_log_likelihood(dshared, segment):
    info_file = dshared['info_file']
    return info_file['segment_' + str(segment)].attrs['log_likelihood'] 

def get_posterior_table(d):
    return d['posterior_table']

def get_dnase_data(d):
    return d['dnase']

def get_n_obs(d):
    return d['n_obs']

def get_n_states(d):
    return d['n_states']

def get_n_tfs(d):
    return d['n_tfs']

def get_nuc_start(d):
    return d['nuc_start']
    
def set_tmpDir(d, tmpDir):
    d['tmpDir'] = tmpDir

def get_segment(d):
    d['segment']

def set_segment(d, segment):
    d['segment'] = segment

