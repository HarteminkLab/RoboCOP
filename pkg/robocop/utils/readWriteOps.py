import pickle
import numpy as np

# save dictionary
def dumpIdx(x, dirname):
    xt = {'segment': x['segment'], 'log_likelihood': x['log_likelihood']}
    with open(dirname + "dict.idx" + str(x['segment']) + ".pkl", "wb") as writeFile:
        pickle.dump(xt, writeFile, pickle.HIGHEST_PROTOCOL)
    np.save(dirname + "posterior_table.idx" + str(x['segment']), x['posterior_table'])
    np.save(dirname + "data_emission_matrix.idx" + str(x['segment']), x['data_emission_matrix'])

# load dictionary for a particular segment
def loadIdx(dirname, segment):
    with open(dirname + "dict.idx" + str(segment) + ".pkl", "rb") as readFile:
        x = pickle.load(readFile)
    x['posterior_table'] = np.load(dirname + "posterior_table.idx" + str(segment) + ".npy", allow_pickle = True)
    x['data_emission_matrix'] = np.load(dirname + "data_emission_matrix.idx" + str(segment) + ".npy", allow_pickle = True)
    return x
    
