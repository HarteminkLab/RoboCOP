import pickle
import numpy as np

# save dictionary
def dumpIdx(x, dirname):
    xt = {'segment': x['segment'], 'log_likelihood': x['log_likelihood']}
    with open(dirname + "dict.idx" + str(x['segment']) + ".pkl", "wb") as writeFile:
        pickle.dump(xt, writeFile, pickle.HIGHEST_PROTOCOL)
    # Saving compressed posterior and data emission matrix
    np.savez_compressed(dirname + "posterior_and_emission.idx" + str(x['segment']), posterior = x['posterior_table'], emission = x['data_emission_matrix'])
    # np.save(dirname + "posterior_table.idx" + str(x['segment']), x['posterior_table'])
    # np.save(dirname + "data_emission_matrix.idx" + str(x['segment']), x['data_emission_matrix'])

# load dictionary for a particular segment
def loadIdx(dirname, segment):
    with open(dirname + "dict.idx" + str(segment) + ".pkl", "rb") as readFile:
        x = pickle.load(readFile)
    # x['posterior_table'] = np.load(dirname + "posterior_table.idx" + str(segment) + ".npy", allow_pickle = True)
    # x['data_emission_matrix'] = np.load(dirname + "data_emission_matrix.idx" + str(segment) + ".npy", allow_pickle = True)
    # load compressed matrix
    d = np.load(dirname + "posterior_and_emission.idx" + str(x['segment']) + ".npz", allow_pickle = True)
    x['posterior_table'] = d['posterior']
    x['data_emission_matrix'] = d['emission']
    return x
    
