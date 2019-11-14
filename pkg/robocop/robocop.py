from ctypes import *
import numpy as np
from numpy.ctypeslib import ndpointer
from scipy.stats import nbinom
import sys, os
from .nucleosome import nuc_dinucleotide_model_file, nuc_model_length
from .utils import parameters as nuc_parameters
import pickle

def reverse_complement(pwm):
    """
    Given a pwm (4 by n numpy array), return the reverse complement pwm
    """
    rc_pwm = np.zeros((4, len(pwm[0,])))
    rc_pwm[0, ] = pwm[3, ::-1]
    rc_pwm[1, ] = pwm[2, ::-1]
    rc_pwm[2, ] = pwm[1, ::-1]
    rc_pwm[3, ] = pwm[0, ::-1]
    return rc_pwm

def createDictionary(segment, dshared):
    """
    create dictionary for one segment and construct and save
    data emission matrix for the segment.
    """
    d = {}
    d['segment'] = segment
    build_data_emission_matrix(d, dshared)
    d['log_likelihood'] = None
    d['posterior_table'] = None
    return d

def createSharedDictionary(d, tf_prob, background_prob, nucleosome_prob, pwmFile, tmpDir, nt_array = None, n_obs = None, lock = None):
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
    check_parameters(d)
    if n_obs != None: d['n_obs'] = n_obs
    else:
        # otherwise read one of the files to get length
        d['nucleotides'] = nt_array
        # assuming all segments are of same size
        if d['nucleotides'] is not None: d['n_obs'] = len(np.load(d['tmpDir'] + "nucleotides.idx0.npy"))
    # build the HMM transition matrix
    build_transition_matrix(d, pwmFile, tf_prob = _tf_prob, background_prob = background_prob, nucleosome_prob = nucleosome_prob, allow_end_at_any_state = 1)
    if nt_array != None: stack_pwms(d, pwmFile)

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

def build_transition_matrix(d, pwmFile, tf_prob, background_prob = 0,
                            nucleosome_prob = 0, allow_end_at_any_state = 1):
    """Build the backbone of the transition matrix. Set transition inside
    motifs to be 1. Set the nucleosome dinucleotide mode if nuc is present."""
    get_transition_matrix_info(d, pwmFile, allow_end_at_any_state)
    _build_transition_matrix(d)
    set_transition(d, [], background_prob, nucleosome_prob)
    set_initial_probs(d)
    set_end_probs(d)
        
def get_transition_matrix_info(d, pwmFile, allow_end_at_any_state):
    """
    Obtain the information needed to construct the transition matrix,
    based on DBFs in the model and PWM used. 
    """
    pwm = pickle.load(open(pwmFile, "rb"), encoding = "latin1")
    nucleosome_prob = d['nucleosome_prob']
    tfs = d['tfs']
    # adding padding to TFs on both sides
    if d['nucleotides'] is None: tf_lens = np.array([(10 + 2*d['padding']) for tf in tfs])
    else: tf_lens = np.array(
        [(pwm[tf]['matrix'].shape[1] + 2*d['padding']) for tf in tfs]) # setting default TF length to 10
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
    d['n_vars'] = 4
    d['silent_states_begin'] = silent_states_begin
    d['tf_starts'] = tf_starts
    d['tf_lens'] = tf_lens

# construct transition matrix
def _build_transition_matrix(d):
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

def stack_pwms(d, pwmFile):
    #background must be in the model
    pwm = pickle.load(open(pwmFile, "rb"), encoding = 'latin1')
    tfs = d['tfs']
    pwm_emat = np.transpose(pwm['background']['matrix'])
    for tf in tfs:
        # add padding -- padding is 0 so no padding
        for i in range(d['padding']): pwm_emat = np.vstack((pwm_emat, np.transpose(pwm['background']['matrix'])))
        pwm_emat = np.vstack((pwm_emat, np.transpose(pwm[tf]['matrix'])))
        # add padding -- padding is 0 so no padding
        for i in range(d['padding']): pwm_emat = np.vstack((pwm_emat, np.transpose(pwm['background']['matrix'])))
        
        # add padding -- padding is 0 so no padding
        for i in range(d['padding']): pwm_emat = np.vstack((pwm_emat, np.transpose(pwm['background']['matrix'])))
        pwm_emat = np.vstack((pwm_emat, np.transpose(
            reverse_complement(pwm[tf]['matrix']))))
        # add padding -- padding is 0 so no padding
        for i in range(d['padding']): pwm_emat = np.vstack((pwm_emat, np.transpose(pwm['background']['matrix'])))


    if d['nuc_present']:
        pwm_emat = np.vstack((pwm_emat, nuc_parameters.nuc_emission))
    assert pwm_emat.shape[0] == d['silent_states_begin']
    _pwm_emat = np.ascontiguousarray(pwm_emat)
    d['pwm_emission'] = _pwm_emat
    
def _build_data_emission_matrix(d, dshared):
    # 3d matrix -- dim 0 is for timepoints, 
    # dim 1 denotes the multivariate emission values
    # right now it's (0) nucleotide, (1) mnase-short, (2) mnase-long, (3) dnase
    # need to remove hard codedness in future
    for t in range(dshared['timepoints']):
        data_emat = np.ones((4, dshared['n_obs'], dshared['n_states']))
        d['data_emission_matrix'] = data_emat
        if dshared['nucleotides'] is not None: update_data_emission_matrix_using_nucleotides(d, dshared)
    
def build_data_emission_matrix(d, dshared):
    _build_data_emission_matrix(d, dshared)

def update_data_emission_matrix_using_nucleotides(d, dshared):
    robocopC = CDLL(dshared["robocopC"])
    robocopC.build_emission_mat_from_pwm.argtypes = [ndpointer(np.long, flags = "C_CONTIGUOUS"), ndpointer(np.double, flags = "C_CONTIGUOUS"), c_int, c_int, c_int, c_int, ndpointer(np.double, flags = "C_CONTIGUOUS")]
    data_emission_matrix = d['data_emission_matrix']
        
    for t in range(dshared['timepoints']):
        nucleotides = np.load(dshared['tmpDir'] + "nucleotides.idx" + str(d['segment']) + ".npy")
        data_emat = data_emission_matrix[0]
        pwm = dshared['pwm_emission']
        
        nucleotides = nucleotides.astype(np.long)
        pwm = pwm.astype(np.double)
        data_emat = data_emat.astype(np.double)

        robocopC.build_emission_mat_from_pwm(nucleotides, pwm, dshared['n_obs'], dshared['n_states'], dshared['silent_states_begin'], 4, data_emat)

        data_emission_matrix[0] = data_emat
        d['data_emission_matrix'] = data_emission_matrix

def update_data_emission_matrix_using_negative_binomial(
            d, dshared, phis, mus, data, index, timepoint):
    """
    Update the data emission matrix based on the negative binomial
    distribution.
    This function allows using different phi and mu for every single state.
    phis: an array containing phi for every non-silent state 
    mus:  an array containing mu for every non-silent state 
    data: an array of integer data_emission_matrix
    """
    data_emission_matrix = d['data_emission_matrix']
    dictionary = {}
    for i in range(dshared['n_obs']):
        for j in range(dshared['silent_states_begin']):
            if (phis[j], mus[j]) not in dictionary:
                dictionary[(phis[j], mus[j])] = {}
                p = phis[j]/(mus[j] + phis[j])
                dictionary[(phis[j], mus[j])][data[i]] = nbinom.pmf(data[i], phis[j], p)
            elif data[i] not in dictionary[(phis[j], mus[j])]:
                p = phis[j]/(mus[j] + phis[j])
                dictionary[(phis[j], mus[j])][data[i]] = nbinom.pmf(data[i], phis[j], p)
            data_emission_matrix[index][i][j] *= dictionary[(phis[j], mus[j])][data[i]]
    d['data_emission_matrix'] = data_emission_matrix

def update_data_emission_matrix_using_mnase_midpoint_counts_onePhi(
            d, dshared, nuc_phi, nuc_mus, tf_phi, tf_mu, other_phi, other_mu, mnaseType):
    """
    Update data emission matrix using N.B.
    """
    tf_starts = dshared['tf_starts']
    tf_lens = dshared['tf_lens']
    if mnaseType == 'short':
        mnaseData = np.load(dshared['tmpDir'] + "MNaseShort.idx" + str(d['segment']) + ".npy")
    else:
        mnaseData = np.load(dshared['tmpDir'] + "MNaseLong.idx" + str(d['segment']) + ".npy")
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
        if mnaseType == 'short': update_data_emission_matrix_using_negative_binomial(d, dshared, phis, mus, mnaseData, 1, t)
        else: update_data_emission_matrix_using_negative_binomial(d, dshared, phis, mus, mnaseData, 2, t)

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
    # check initial probs
    assert np.abs(np.sum(initial_probs) - 1) < 0.00001
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
    robocopC = CDLL(dshared["robocopC"])
    motif_starts = dshared['tf_starts'] 
    motif_lens = dshared['tf_lens'] 
    initial_probs = dshared['initial_probs']
    end_probs = dshared['end_probs']
    transition_mat = dshared['transition_matrix']
    data_emission_mat = d['data_emission_matrix']
    fscaling_factors = np.zeros(dshared['n_obs'])
    bscaling_factors = np.zeros(dshared['n_obs'])
    scaling_factors = np.zeros(dshared['n_obs'])
    parents_mat = np.zeros((dshared['n_states'], dshared['n_states']), dtype = int)
    n_parents = np.zeros(dshared['n_states'], dtype = int)
    children_mat = np.zeros((dshared['n_states'], dshared['n_states']), dtype = int)
    n_children = np.zeros(dshared['n_states'], dtype = int)
    ftable = np.zeros((dshared['n_obs'], dshared['n_states']))
    btable = np.zeros((dshared['n_obs'], dshared['n_states']))
    p_table = np.zeros((dshared['n_obs'], dshared['n_states']))
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
    scaling_factors = scaling_factors.astype(np.double)
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
        dshared['n_states'], dshared['silent_states_begin'], dshared['n_obs'], dshared['n_vars'], 
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
        dshared['n_states'], dshared['silent_states_begin'], dshared['n_obs'], dshared['n_vars'],
        children_mat, n_children,
        motif_starts, motif_lens, dshared['n_tfs'],
        dshared['nuc_present'], dshared['nuc_start'], dshared['nuc_len'],
        btable, bscaling_factors)

    robocopC.calc_sr.argtypes = [ndpointer(np.double, flags = "C_CONTIGUOUS"), ndpointer(np.double, flags = "C_CONTIGUOUS"), c_int, ndpointer(np.double, flags = "C_CONTIGUOUS")]
    robocopC.calc_sr.restype = c_double
    log_fscaling_factor_sum_loop = robocopC.calc_sr(
        fscaling_factors, bscaling_factors,
        dshared['n_obs'], scaling_factors)

    # posterior decoding
    robocopC.posterior_decoding.argtypes = [ndpointer(np.double, flags = "C_CONTIGUOUS"), ndpointer(np.double, flags = "C_CONTIGUOUS"), ndpointer(np.double, flags = "C_CONTIGUOUS"), ndpointer(np.double, flags = "C_CONTIGUOUS"), c_int, c_int, ndpointer(np.double, flags = "C_CONTIGUOUS")]
    robocopC.posterior_decoding(
        ftable, btable, bscaling_factors,
        scaling_factors, dshared['n_states'], dshared['n_obs'], 
        p_table)
    d['posterior_table'] = p_table
    return log_fscaling_factor_sum_loop

def posterior_forward_backward(d, dshared, lock = None):
    log_fscaling_factor_sum = 0
    for t in range(dshared['timepoints']):
        log_fscaling_factor_sum += posterior_forward_backward_loop(d, dshared, lock)
    d['log_likelihood'] = log_fscaling_factor_sum


def viterbi_decoding(d, dshared):
    """
    Forward backward and posterior decoding.
    """
    robocopC = CDLL(dshared["robocopC"])
    motif_starts = dshared['tf_starts'] 
    motif_lens = dshared['tf_lens'] 
    initial_probs = dshared['initial_probs']
    end_probs = dshared['end_probs']
    transition_mat = dshared['transition_matrix']
    data_emission_mat = d['data_emission_matrix']
    fscaling_factors = np.zeros(dshared['n_obs'])
    bscaling_factors = np.zeros(dshared['n_obs'])
    scaling_factors = np.zeros(dshared['n_obs'])
    parents_mat = np.zeros((dshared['n_states'], dshared['n_states']), dtype = int)
    n_parents = np.zeros(dshared['n_states'], dtype = int)
    children_mat = np.zeros((dshared['n_states'], dshared['n_states']), dtype = int)
    n_children = np.zeros(dshared['n_states'], dtype = int)
    vtable = np.zeros((dshared['n_obs'], dshared['n_states']))
    vpointer = np.zeros((dshared['n_obs'], dshared['n_states']))
    parents_mat = parents_mat.astype(np.long)
    children_mat = children_mat.astype(np.long)
    n_parents = n_parents.astype(np.long)
    n_children = n_children.astype(np.long)
    transition_mat = transition_mat.astype(np.double)
    data_emission_mat = data_emission_mat.astype(np.double)
    initial_probs = initial_probs.astype(np.double)
    end_probs = end_probs.astype(np.double)
    vtable = vtable.astype(np.double)
    vpointer = vpointer.astype(np.double)
    fscaling_factors = fscaling_factors.astype(np.double)
    bscaling_factors = bscaling_factors.astype(np.double)
    scaling_factors = scaling_factors.astype(np.double)
    motif_starts = motif_starts.astype(np.long)
    motif_lens = motif_lens.astype(np.long)
    
    robocopC.find_parents_and_children.argtypes = [ndpointer(np.long, flags = "C_CONTIGUOUS"), ndpointer(np.long, flags = "C_CONTIGUOUS"), ndpointer(np.long, flags = "C_CONTIGUOUS"), ndpointer(np.long, flags = "C_CONTIGUOUS"), c_int, c_int, ndpointer(np.double, flags = "C_CONTIGUOUS")]
    robocopC.find_parents_and_children(parents_mat, children_mat,
                                         n_parents, n_children, dshared['n_states'],
                                         dshared['silent_states_begin'], transition_mat)

    # viterbi algorithm
    robocopC.viterbi.argtypes = [ndpointer(np.double, flags = "C_CONTIGUOUS"), ndpointer(np.double, flags = "C_CONTIGUOUS"), ndpointer(np.double, flags = "C_CONTIGUOUS"), ndpointer(np.double, flags = "C_CONTIGUOUS"), c_int, c_int, c_int, c_int, ndpointer(np.long, flags = "C_CONTIGUOUS"), ndpointer(np.long, flags = "C_CONTIGUOUS"), ndpointer(np.long, flags = "C_CONTIGUOUS"), ndpointer(np.long, flags = "C_CONTIGUOUS"), c_int, c_int, c_int, c_int, ndpointer(np.double, flags = "C_CONTIGUOUS"), ndpointer(np.double, flags = "C_CONTIGUOUS")]
    robocopC.viterbi(
        initial_probs,
        transition_mat, data_emission_mat,
        end_probs,
        dshared['n_states'], dshared['silent_states_begin'], dshared['n_obs'], dshared['n_vars'], 
        parents_mat, n_parents,
        motif_starts, motif_lens, dshared['n_tfs'],
        dshared['nuc_present'], dshared['nuc_start'], dshared['nuc_len'],
        vtable, vpointer
    )
    d['viterbi_table'] = vtable
    d['viterbi_traceback'] = vpointer
    return 0


def center_for_dbf_probs(d, dshared): #, lock = None):
    """
    the length of nucleosome padding is defined here as 0, for now
    This may need to re implemented in C for speed 
    """
    
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
        dbf_binding_probs = np.zeros((dshared['n_obs'], dshared['n_tfs'] + 1 + 5*dshared['nuc_present']))
        posterior_table = d['posterior_table']
        for i in range(dshared['n_obs']):
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

def sum_for_dbf_probs(d, dshared):
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
        dbf_binding_probs = np.zeros((dshared['n_obs'], dshared['n_tfs'] + 1 + 5*dshared['nuc_present']))
        posterior_table = d['posterior_table']
        for i in range(dshared['n_obs']):
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
            if dbf_binding_probs[i, dshared['n_tfs'] + 2] > 1.000000001: print("GREATER: ", dbf_binding_probs[t, i, dshared['n_tfs'] + 2])
        return dbf_binding_probs

# save posterior probability as csv
def print_posterior_binding_probability(d, dshared, file_name = '', 
                                        by_dbf = True, just_center = False):
    tfs = dshared['tfs']
    for t in range(dshared['timepoints']):
        posterior_table = d['posterior_table']
        if file_name == '':
            from sys import stdout 
            output = stdout 
        else:
            output = open(file_name + '.timepoint' + str(d['segment']), 'w')
        if not by_dbf:
            for i in range(dshared['n_obs']):
                output.write(
                    '\t'.join(['%g' % _ for _ in d['posterior_table'][i,]]) + '\n'
                )
        else:
            if just_center: dbf_binding_probs = center_for_dbf_probs(d, dshared)
            else: dbf_binding_probs = sum_for_dbf_probs(d, dshared)
            header = "background"
            if dshared['n_tfs'] > 0:
                header += "\t%s" % '\t'.join(tfs)
            if dshared['nuc_present']:
                header += "\tnuc_padding\tnucleosome\tnuc_center\tnuc_start\tnuc_end"
            output.write(header + '\n')
            for i in range(dshared['n_obs']):
                output.write('\t'.join(['%g' % _ for _ in dbf_binding_probs[i, ]]) + '\n')
        if file_name != '':
            output.close()
        


############################################
## Functions for extracting internal data ##
############################################

def get_states_info(d):
    return d['n_states'], d['silent_states_begin']

def get_initial_probs(d):
    return d['initial_probs'] 

def get_pwm_emission(d):
    return d['pwm_emission']

def get_log_likelihood(d):
    return d['log_likelihood']

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

