"""
Take the data frame generated using function get_posterior_binding_probability_df()
and plot the binding landscape. Plots on the given axis handle
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

########### some internal configurations ##############
annotate_ymin = -0.2 
annotate_ymax = -0.05 
ymax = 1.1 # the y limit 


def visualize_dbf_color_map(dbf_color_map):
    plt.figure(figsize=(10,10))
    dbfs = sorted(dbf_color_map.keys())
    ndbfs = len(dbfs)
    ys = np.arange(0, ndbfs * 20, 20)

    plt.hlines(y = ys, xmin=0, xmax=10, colors = [dbf_color_map[_] for _ in dbfs], linewidth=10);
    plt.ylim(-15, ys[-1] + 15)
    plt.yticks(ys, dbfs);

def preprocess_occupancy_profile(op, drop_threshold):
    '''process occupancy profile before plotting'''
    tfs = [x for x in list(op) if x != "nucleosome" and x != "nuc_start" and x != "nuc_center" and x != "unknown" and x != "background" and x != "nuc_padding" and x != "nuc_end"]
    combinedTFs = list(set([(x.split("_")[0]).upper() for x in tfs]))
    newOP = pd.DataFrame(columns = combinedTFs + ["nucleosome", "nuc_start", "nuc_center", "unknown", "coordinate"])
    if "unknown" in list(op): newOP["unknown"] = op["unknown"]
    else: newOP["unknown"] = [0 for i in range(len(op))]
    newOP["nucleosome"] = op["nucleosome"]
    newOP["nuc_start"] = op["nuc_start"]
    newOP["nuc_center"] = op["nuc_center"]
    newOP["coordinate"] = op["coordinate"]
    for tf in combinedTFs:
        tfSet = [x for x in tfs if (x.split("_")[0]).upper() == tf]
        newOP[tf] = np.sum(op[tfSet], axis = 1)
        
    # #drop columns that have low binding prob
    drop_columns = set(newOP.columns[newOP.apply(max) < drop_threshold])
    newOP = newOP.drop(drop_columns, axis = 1)

    # replace all values less than threshold with 0
    for col in list(newOP): newOP[col] = np.where(newOP[col] < drop_threshold, 0, newOP[col])
    return newOP

def plot_dbf_binding(op, dbf_color_map, nucDyad, ax):
    #separate nucleosome from other DBFs
    dbfs = list(op.columns.values)
    dbfs.remove('coordinate')
    if 'nucleosome' in dbfs:
        dbfs.remove('nucleosome')
        if 'nuc_center' in dbfs: dbfs.remove('nuc_center')
        if 'nuc_start' in dbfs: dbfs.remove('nuc_start')
        nuc_present = True
        
    else:
        nuc_present = False

    #plot nucleosome first
    if nuc_present:
        if nucDyad:
            ax.plot(op.coordinate, op.loc[:, 'nuc_center'], color = dbf_color_map['nucleosome'], label = 'nuc_dyad')
            ax.fill_between(op.coordinate, op.loc[:, 'nuc_center'], color = dbf_color_map['nucleosome'])
            ax.plot(op.coordinate, op.loc[:, 'nuc_start'], color = '#000000', label = 'nuc_start')
            ax.fill_between(op.coordinate, op.loc[:, 'nuc_start'], color = '#000000')
        else:
            ax.plot(op.coordinate, op.loc[:, 'nucleosome'], color = dbf_color_map['nucleosome'], label = 'nuc')
            ax.fill_between(op.coordinate, op.loc[:, 'nucleosome'], color = dbf_color_map['nucleosome'])


    # plot unknown first
    dbf = 'unknown'
    if dbf in list(op):
        ax.plot(op.coordinate, op.loc[:, dbf], color = dbf_color_map[dbf], label = dbf, alpha = 0.5)
        ax.fill_between(op.coordinate, op.loc[:, dbf], color = dbf_color_map[dbf], alpha = 0.5)


    #plot all other dbfs
    for dbf in dbfs:
        if dbf == 'unknown': continue
        if dbf not in list(op): continue
        if dbf not in list(dbf_color_map): continue
        ax.plot(op.coordinate, op.loc[:, dbf], color = dbf_color_map[dbf], label = dbf)
        ax.fill_between(op.coordinate, op.loc[:, dbf], color = dbf_color_map[dbf])

        
def plot_occupancy_profile(ax, op, chromo, coordinate_start, dbf_color_map, padding = 0, threshold = 0.1, plot_legend = True, figsize=(18,4), file_name = None, nucDyad = False):

        
    op['coordinate'] = np.arange(coordinate_start, coordinate_start + op.shape[0])
    op = op.iloc[padding:(-padding-1), :]
    
    op = preprocess_occupancy_profile(op, drop_threshold=threshold)

    #plot DBFs using area plot
    plot_dbf_binding(op, dbf_color_map, nucDyad, ax)
    
    #######################  set axis properties #######################
    ax.set_ylim(0, 1)


    if plot_legend:
        # legend
        leg = ax.legend(loc='lower left', ncol = 12, bbox_to_anchor = (0., 1.),
                        borderaxespad=0, frameon=False, framealpha=0)
        # for legobj in ax.get_legend_handles_labels():
        #     legobj.set_linewidth(10.0)
        # if leg != None:
        #     for legobj in leg.legendHandles:
        #         legobj.set_linewidth(10.0)

    #set the y axis bounds so only the 0-1.0 part is shown
    ax.spines['left'].set_bounds(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1]);

    # delete the x axis, but the ticks will remain
    ax.spines['bottom'].set_linewidth(0)

    # remove top and right borders   
    ax1 = plt.gca()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')

