import h5py
import pickle
import numpy as np

# save dictionary
def dumpIdx(x, info_file): # dirname):
    k = "segment_" + str(x['segment'])
    if k not in info_file.keys():
        g = info_file.create_group(k)
    else:
        g = info_file[k]
    g.attrs['segment'] = x['segment']
    if x['log_likelihood'] is not None: g.attrs['log_likelihood'] = x['log_likelihood']
    g.attrs['chr'] = x['chr']
    g.attrs['start'] = x['start']
    g.attrs['end'] = x['end']
    g.attrs['n_obs'] = x['n_obs']

