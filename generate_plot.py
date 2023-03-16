#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate plots by calling the plot_1D function in the main file. 
This includes plotting: 
    potential across time at a particular cell (plot_1D_cell),
    potential across cells (spatially) at a particular time (plot_1D_array),
    and potential across time and space (plot_1D_grid).
The function takes inputs:
    data_list, dynamics, model, fig_name, x_t_to_plot, i.

Commented by Annie
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

def plot_1D(data_list, dynamics, model, fig_name, x_t_to_plot, i):

    plot_1D_cell(data_list, dynamics, model, fig_name[1:], x_t_to_plot[0], i)
    plot_1D_array(data_list, dynamics, model, fig_name[1:], x_t_to_plot[1], i)
    plot_1D_grid(data_list, dynamics, model, fig_name[1:], x_t_to_plot, i)
    return 0
    
def plot_1D_cell(data_list, dynamics, model, fig_name, x_to_plot, i):
    
     ## Unpack data
    observe_x, observe_train, v_train, v, observe_test, v_test= \
          data_list[0], data_list[1], data_list[2], data_list[3], data_list[4], data_list[5]
    
    ## Pick a cell to show
    #cell = dynamics.max_x*x_to_plot
    cell = x_to_plot
    lnw = 3.0 # line width
    szm = 50 # marker size
    ftsz = 20 # font size
    
    ## Get data for cell
    idx = [i for i,ix in enumerate(observe_x) if observe_x[i][0]==cell]
    observe_geomtime = observe_x[idx]
    v_GT = v[idx]
    v_predict = model.predict(observe_geomtime)[:,0:1]
    t_axis = observe_geomtime[:,1]
    
    ## Get data for points used in training process
    idx_train = [i for i,ix in enumerate(observe_train) if observe_train[i][0]==cell]
    v_trained_points = v_train[idx_train]
    t_markers = (observe_train[idx_train])[:,1]

    ## Get data for points used in testing
    idx_test = [i for i,ix in enumerate(observe_test) if observe_test[i][0]==cell]
    v_test_points = v_test[idx_test]
    t_markers_test = (observe_test[idx_test])[:,1]
    
    ## create figure for cell
    plt.figure()
    plt.rc('font', size= ftsz) #controls default text
    fig, ax = plt.subplots()
    Predicted, = ax.plot(t_axis, v_predict, c='r', label='Predicted',linewidth=lnw, zorder=0)
    GT, = ax.plot(t_axis, v_GT, c='b', label='GT',linewidth=lnw, linestyle = 'dashed', zorder=5)
    #plt.plot(t_axis, v_GT, c='b', label='GT')
    #plt.plot(t_axis, v_predict, c='r', label='Predicted')
    # If there are any trained data points for the current cell 
    if len(t_markers):
        ax.scatter(t_markers, v_trained_points, marker='x', c='black',s=szm, label='Training points', zorder=10)
    plt.legend(loc='upper right', borderpad=0.2)
    #if there are any test points for the current cell
    if len(t_markers_test):
        ax.scatter(t_markers_test, v_test_points, marker='x', c='green',s=szm, label='Testing points',zorder=10)

    plt.title("_run_" + str(i) + "cell_at" + str(x_to_plot) + "mm")
    plt.xlabel('t (TU)', fontsize = ftsz)
    plt.ylabel('u (AU)', fontsize = ftsz)
    plt.ylim((-0.2,1.2))
    
    ## save figure
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=500, pad_inches = .1, bbox_inches = 'tight')
    png2 = Image.open(png1)
    png2.save("run_" + str(i) + "cell_at" + str(x_to_plot) + "mm_plot_1D.tiff")
    png1.close()
    return 0

def plot_1D_array(data_list, dynamics, model, fig_name, t_to_plot, i):
    
    ## Unpack data
    observe_x, observe_train, v_train, v = data_list[0], data_list[1], data_list[2], data_list[3]
    
    ## Pick a point in time to show
    obs_t = dynamics.max_t*t_to_plot
    lnw = 3.0 # line width
    szm = 26 # marker size
    ftsz = 20 # font size    
    
    ## Get all array data for chosen time 
    idx = [i for i,ix in enumerate(observe_x) if observe_x[i][1]==obs_t]
    observe_geomtime = observe_x[idx]
    v_GT = v[idx]
    v_predict = model.predict(observe_geomtime)[:,0:1]
    x_ax = observe_geomtime[:,0]
    
    ## Get data for points used in training process
    idx_train = [i for i,ix in enumerate(observe_train) if observe_train[i][1]==obs_t]
    v_trained_points = v_train[idx_train]
    x_markers = (observe_train[idx_train])[:,0]

    ## create figure
    plt.figure()
    plt.rc('font', size= ftsz) #controls default text size
    plt.plot(x_ax, v_predict, c='r', label='Predicted',linewidth=lnw)
    plt.plot(x_ax, v_GT, c='b', label='GT',linewidth=lnw, linestyle = 'dashed')
    # If there are any trained data points for the current time step
    if len(x_markers):
        plt.scatter(x_markers, v_trained_points, marker='x', c='black',s=szm, label='Observed')
    plt.legend(fontsize = ftsz, loc = 'lower center')
    plt.xlabel('x (mm)', fontsize = ftsz)
    plt.ylabel('u (AU)', fontsize = ftsz)
    plt.ylim((-0.2,1.2))
    
    ## save figure
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=500, pad_inches = .1, bbox_inches = 'tight')
    png2 = Image.open(png1)
    png2.save("run_" + str(i) + "_array_plot_1D.tiff")
    png1.close()
    return 0

def plot_1D_grid(data_list, dynamics, model, fig_name, x_t_to_plot, i):
    
    grid_size = 200
    
    ## Get data
    x = np.linspace(dynamics.min_x,dynamics.max_x, grid_size)
    t = np.linspace(dynamics.min_t,dynamics.max_t,grid_size)
    X, T = np.meshgrid(x,t)
    X_data = X.reshape(-1,1)
    T_data = T.reshape(-1,1)
    data = np.hstack((X_data, T_data))
    v_pred = model.predict(data)[:,0:1]
    Z = np.zeros((grid_size,grid_size))
    for i in range(grid_size):
        Z[i,:] = (v_pred[(i*grid_size):((i+1)*grid_size)]).reshape(-1)
    
    ## create figure
    plt.figure()
    contour = plt.contourf(T,X,Z, levels = np.arange(-0.15,1.06,0.15) , cmap=plt.cm.bone)
    plt.plot(t, np.full(len(t), 2), '--', label = 'slice across t')
    plt.vlines(dynamics.max_t * x_t_to_plot[1], ymin = 0, ymax = dynamics.max_x,
           colors = 'purple', linestyles = '--', label = 'slice across x')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend()
    cbar = plt.colorbar(contour)
    cbar.ax.set_ylabel('Membrane Potential U (AU)')
    
    ## save figure
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=500, pad_inches = .1, bbox_inches = 'tight')
    png2 = Image.open(png1)
    png2.save("run_" + str(i) + "_grid_plot_1D.tiff")
    png1.close()
    
    return 0
    
    return 0