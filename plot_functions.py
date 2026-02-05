import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.spatial import KDTree
import ast
import seaborn as sns
import time
import os
from multiprocessing import Pool
import requests
import ast
from matplotlib.colors import LogNorm
import pickle
import scipy.sparse as sp
from scipy.signal import convolve2d

bri_names_1d = ['x(N)', 'y(N)', 'z(N)', 'x(A)', 'y(A)', 'z(A)', 'x(C)', 'y(C)', 'z(C)']
bri_names_1d_latex = ['$x_{BRI}(C_{i-1}N_{i})$','$y_{BRI}(C_{i-1}N_{i})$', '$z_{BRI}(C_{i-1}N_{i})$',
                '$x_{BRI}(N_{i}A_{i})$', '$y_{BRI}(N_{i}A_{i})$','$z_{BRI}(N_{i}A_{i})$',
                '$x_{BRI}(A_{i}C_{i})$', '$y_{BRI}(A_{i}C_{i})$', '$z_{BRI}(A_{i}C_{i})$']

bri_input_parameters_1d = [(0.01,-2.0,2.0),(0.01,-2.0,2.0),(0.01,-2.0,2.0),
                           (0.01,-2.0,2.0),(0.01,-2.0,2.0),(0.01,-2.0,2.0),
                           (0.01,-2.0,2.0),(0.01,-2.0,2.0),(0.01,-2.0,2.0)]

bri_names_2d = []
bri_names_2d_latex = []
bri_input_parameters_2d = []
for i in [(0,1),(0,2),(1,2),(3,4),(3,5),(4,5),(6,7),(6,8),(7,8)]:
    bri_names_2d.append((bri_names_1d[i[0]],bri_names_1d[i[1]]))
    bri_names_2d_latex.append((bri_names_1d_latex[i[0]],bri_names_1d_latex[i[1]]))
    bri_input_parameters_2d.append((0.01,0.01,-2.0,2.0,-2.0,2.0))


trin_names_1d = ['x(AN)', 'x(AC)', 'y(AC)']
trin_names_1d_latex = ['$x_{TRIN}(A_{i}N_{i})$', '$x_{TRIN}(A_{i}C_{i})$', '$y_{TRIN}(A_{i}C_{i})$']

trin_input_parameters_1d = [(0.01,-3.0,3.0),(0.01,-3.0,3.0),(0.01,-3.0,3.0)]

trin_names_2d = []
trin_names_2d_latex = []
trin_input_parameters_2d = []
for i in [(0,1),(0,2),(1,2)]:
    trin_names_2d.append((trin_names_1d[i[0]],trin_names_1d[i[1]]))
    trin_names_2d_latex.append((trin_names_1d_latex[i[0]],trin_names_1d_latex[i[1]]))
    trin_input_parameters_2d.append((0.01,0.01,-3.0,3.0,-3.0,3.0))

al_names_1d = ['length(N)', 'length(A)', 'length(C)', 'angle(N)', 'angle(A)', 'angle(C)', 'tau(NA)', 'tau(AC)', 'tau(CN)']

al_names_1d_latex = ['$len(C_{i-1}N_{i})$', '$len(N_{i}A_{i})$', '$len(A_{i}C_{i})$',
                     '$angle(C_{i-1}N_{i}A_{i})$', '$angle(N_{i}A_{i}C_{i})$','$angle(A_{i}C_{i}N_{i+1})$',
                     '$tau(C_{i-1}N_{i}A_{i},N_{i}A_{i}C_{i})$', '$tau(N_{i}A_{i}C_{i},A_{i}C_{i}N_{i+1})$','$tau(A_{i-1}C_{i-1}N_{i},C_{i-1}N_{i}A_{i})$']

al_input_parameters_1d = [(0.01,0.0,3.0),(0.01,0.0,3.0),(0.01,0.0,3.0),
 (0.1,0.0,180.0),(0.1,0.0,180.0),(0.1,0.0,180.0),
  (0.1,-180.0,180.0),(0.1,-180.0,180.0),(0.1,-180.0,180.0)]

al_names_2d = []
al_names_2d_latex = []
al_input_parameters_2d = []
for i in [(0,1),(0,2),(1,2),
          (3,4),(3,5),(4,5),
          (6,7),(6,8),(7,8)]:

    al_names_2d.append((al_names_1d[i[0]],al_names_1d[i[1]]))
    al_names_2d_latex.append((al_names_1d_latex[i[0]],al_names_1d_latex[i[1]]))
    if i in [(0,1),(0,2),(1,2)]:
        al_input_parameters_2d.append((0.01,0.01,0.0,3.0,0.0,3.0))
    elif i in [(3,4),(3,5),(4,5)]:
        al_input_parameters_2d.append((0.1,0.1,0.0,180.0,0.0,180.0))
    elif i in [(6,7),(6,8),(7,8)]:
        al_input_parameters_2d.append((0.1,0.1,-180.0,180.0,-180.0,180.0))
    else:
        al_input_parameters_2d.append((0.01,0.01,-1.0,1.0,-1.0,1.0))



def L_inf(matrix1,matrix2):
    return np.max(np.absolute(matrix1)-np.absolute(matrix2)).round(3)

def L_2(matrix1,matrix2):
    return np.linalg.norm(matrix1-matrix2)

    
# @title Function to shift the ranges of the tau invariants in order to avoid discontinuities

def shift_ranges(al,tau_na_range=[0,360],tau_ac_range=[-135,225],tau_cn_range=[-90,270],tau_co_range=[0,360]):

    hist_lai_overall_2d, lai_xedges_2d, lai_yedges_2d, hist_lai_overall_1d, lai_xedges_1d = al[0].copy(),al[1].copy(),al[2].copy(),al[3].copy(),al[4].copy()

    for i in range(6,9):

        for j in range(0,20):

            idx = i*20 + j

            if i==6:

                x_prime = tau_na_range

            elif i==7:

                x_prime = tau_ac_range

            elif i==8:

                x_prime = tau_cn_range

            sp_xprime = int((x_prime[0]+180)*10)
            lai_xedges_1d[idx] = np.arange(x_prime[0],x_prime[1]+0.1,0.1)
            hist_lai_overall_1d[idx] = np.concatenate([hist_lai_overall_1d[idx][sp_xprime:],hist_lai_overall_1d[idx][0:sp_xprime]])


    for i in range(6,9):

        for j in range(0,20):

            idx = i*20 + j

            if i==6:

                x = tau_na_range
                y = tau_ac_range

            elif i==7:

                x = tau_na_range
                y = tau_cn_range


            elif i==8:

                x = tau_ac_range
                y = tau_cn_range


            sp_x = int((x[0]+180)*10)
            sp_y = int((y[0]+180)*10)

            lai_xedges_2d[idx] = np.arange(x[0],x[1]+0.1,0.1)
            lai_yedges_2d[idx] = np.arange(y[0],y[1]+0.1,0.1)

            new_count =np.zeros([3600,3600])

            hist_lai_matrix = hist_lai_overall_2d[idx].toarray().copy()

            new_count[0:(3600-sp_x),0:(3600-sp_y)] = hist_lai_matrix[sp_x:,sp_y:]
            new_count[0:(3600-sp_x),(3600-sp_y):] = hist_lai_matrix[sp_x:,:sp_y]
            new_count[(3600-sp_x):,0:(3600-sp_y)] = hist_lai_matrix[:sp_x,sp_y:]
            new_count[(3600-sp_x):,(3600-sp_y):] = hist_lai_matrix[:sp_x,:sp_y]

            hist_lai_overall_2d[idx] = sp.csr_matrix(new_count)


    al_output = [hist_lai_overall_2d, lai_xedges_2d, lai_yedges_2d, hist_lai_overall_1d, lai_xedges_1d]

    return(al_output)
    

# Create heatmap plots    
def create_heatmap_plot(data,xedges,yedges,xticks,yticks,name,name_latex,output_folder,data_type,invariant,
                        log_scale=True,additional="",additional_label="",fs=30):

    fig, ax = plt.subplots()
    fig.set_figwidth(12)
    fig.set_figheight(12)

    sns.set(font_scale=2.5)
    sns.set_style("whitegrid")

    if log_scale:
        sns.heatmap(data, cmap='rainbow', cbar=True, norm=LogNorm(), square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    else:
        data[data==0] = np.nan
        sns.heatmap(data, cmap='rainbow', cbar=True, square=True, cbar_kws={"shrink": 0.8}, ax=ax)

    #min, max and spacing
    x_ticks = np.arange(xticks[0],xticks[1]+xticks[2],xticks[2])
    y_ticks = np.arange(yticks[0],yticks[1]+yticks[2],yticks[2])

    xedges_min, yedges_min = min(xedges), min(yedges)
    xedges_gap, yedges_gap = xedges[1]-xedges[0], yedges[1]-yedges[0]

    xticks_idx = np.arange(int(np.round((xticks[0]-xedges_min)/xedges_gap,1)),int(np.round((xticks[1]-xedges_min)/xedges_gap,1))+int(round(xticks[2]/xedges_gap,1)),int(round(xticks[2]/xedges_gap,1)))
    yticks_idx = np.arange(int(np.round((yticks[0]-yedges_min)/yedges_gap,1)),int(np.round((yticks[1]-yedges_min)/yedges_gap,1))+int(round(yticks[2]/yedges_gap,1)),int(round(yticks[2]/yedges_gap,1)))

    plt.xticks(ticks=xticks_idx, labels=x_ticks,rotation=90)
    plt.yticks(ticks=yticks_idx, labels=y_ticks,rotation=0)

    if data_type=='angle':
        plt.xlabel(name_latex[0] + ", degrees{}".format(additional_label),fontsize=fs)
        plt.ylabel(name_latex[1] + ", degrees{}".format(additional_label),fontsize=fs)

    elif data_type=='length':
        plt.xlabel(name_latex[0] + ", Angstroms{}".format(additional_label),fontsize=fs)
        plt.ylabel(name_latex[1] + ", Angstroms{}".format(additional_label),fontsize=fs)
    else:
        plt.xlabel(name_latex[0] + "{}".format(additional_label),fontsize=fs)
        plt.ylabel(name_latex[1] + "{}".format(additional_label),fontsize=fs)

    plt.gca().invert_yaxis()
    plt.tight_layout()

    if log_scale:
        plt.savefig(output_folder+"PDB727K_{}_heatmap_{}_vs_{}_log{}".format(invariant,name[0],name[1],additional)+".PNG",facecolor='white',transparent=False,bbox_inches="tight")
    else:
        plt.savefig(output_folder+"PDB727K_{}_heatmap_{}_vs_{}{}".format(invariant,name[0],name[1],additional)+".PNG",facecolor='white',transparent=False,bbox_inches="tight")
    plt.show()
    
    
def plot_heatmaps(histograms,idx,output_folder,output_folder_log,invariant='AL',shift=False):


    #al pair plots
    if invariant=='AL':

        if shift:
            histograms_shifted = shift_ranges(histograms)
            for i in idx:
                
                xedges = histograms_shifted[1][int(20*i)]
                yedges = histograms_shifted[2][int(20*i)]
                xticks = [int(xedges.min()),int(xedges.max()),90]
                yticks = [int(yedges.min()),int(yedges.max()),90]
                name = al_names_2d[i]
                name_latex = al_names_2d_latex[i]
                data_type='angle'
                data = np.sum(histograms_shifted[0][(20*i):(20*(i+1))]).toarray().T



                create_heatmap_plot(data,xedges,yedges,xticks,yticks,name,name_latex,output_folder,data_type,invariant,
                                log_scale=False,additional="_range_shifted",fs=30)

                create_heatmap_plot(data,xedges,yedges,xticks,yticks,name,name_latex,output_folder_log,data_type,invariant,
                                log_scale=True,additional="_range_shifted",fs=30)

        for i in idx:

            xedges = histograms[1][int(20*i)]
            yedges = histograms[2][int(20*i)]
            name = al_names_2d[i]
            name_latex = al_names_2d_latex[i]
            data = np.sum(histograms[0][(20*i):(20*(i+1))]).toarray().T

            if i in [0,1,2]:
                data = data[75:200,75:200].copy()
                xedges = xedges[75:201]
                yedges = yedges[75:201]
                xticks = [0.75,2.00,0.25]
                yticks = [0.75,2.00,0.25]
                data_type = 'length'
            elif i in [3,4,5]:
                xticks = [0,180,45]
                yticks = [0,180,45]
                data_type = 'angle'
            elif i in [6,7,8]:
                xticks = [-180,180,90]
                yticks = [-180,180,90]
                data_type = 'angle'
            else:
                xticks = [-1.0,1.0,0.5]
                yticks = [-1.0,1.0,0.5]
                data_type = 'sin/cos'

            create_heatmap_plot(data,xedges,yedges,xticks,yticks,name,name_latex,output_folder,data_type,invariant,
                                log_scale=False,additional="",fs=30)

            create_heatmap_plot(data,xedges,yedges,xticks,yticks,name,name_latex,output_folder_log,data_type,invariant,
                                log_scale=True,additional="",fs=30)


    if invariant=='BRI' or invariant=='TRIN':

        if invariant=='BRI':
            names_2d = bri_names_2d
            names_2d_latex = bri_names_2d_latex
        else:
            names_2d = trin_names_2d
            names_2d_latex = trin_names_2d_latex

        for i in idx:

            xedges = histograms[1][int(20*i)]
            yedges = histograms[2][int(20*i)]
            name = names_2d[i]
            name_latex = names_2d_latex[i]
            data = np.sum(histograms[0][(20*i):(20*(i+1))]).toarray().T
            data_type = 'length'
            xticks = [-2,2,0.5]
            yticks = [-2,2,0.5]

            create_heatmap_plot(data,xedges,yedges,xticks,yticks,name,name_latex,output_folder,data_type,invariant,
                                log_scale=False,additional="",fs=30)

            create_heatmap_plot(data,xedges,yedges,xticks,yticks,name,name_latex,output_folder_log,data_type,invariant,
                                log_scale=True,additional="",fs=30)
                                
                                
                                
# GRID HEATMAPS

def lower_matrix_resolution(xedges,yedges,matrix,factor):

    kernel = np.ones((factor, factor))
    convolved = convolve2d(matrix, kernel, mode='valid')
    strided = convolved[::factor, ::factor]

    return(xedges[::factor],yedges[::factor],strided)

def grid_heatmap_plot(data,xedges,yedges,tick_gaps,name,name_latex,output_folder,data_type,invariant,factor,angle=True,log_scale=True,additional="",additional_label=""):

    # Plot the histogram using plt.hist with pre-computed bins and counts
    fig, axes = plt.subplots(5,4)
    fig.set_figwidth(16)
    fig.set_figheight(20)

    sns.set(font_scale=2)
    sns.set_style("whitegrid")

    different_residue_names = ['ALA','CYS','ASP','GLU','PHE','GLY','HIS','ILE','LYS','LEU','MET','ASN','PRO','GLN','ARG','SER','THR','VAL','TRP','TYR']

    # Iterate over the axes and plot the histograms
    for i, ax in enumerate(axes.flat):
        xedges_for_use,yedges_for_use,data_for_use = lower_matrix_resolution(xedges,yedges,data[i].toarray(),factor)
        if not(log_scale):
            data_for_use[data_for_use==0] = np.nan
        pd_df = pd.DataFrame(data_for_use,columns=np.round(xedges_for_use[:-1],2),index=np.round(yedges_for_use[:-1],2))
        if log_scale:
            hm = sns.heatmap(pd_df, cmap='rainbow', cbar=True, norm=LogNorm(), square=True, cbar_kws={"shrink": 0.8}, ax=ax,yticklabels=tick_gaps,xticklabels=tick_gaps)
        else:
            hm = sns.heatmap(pd_df, cmap='rainbow', cbar=True, square=True, cbar_kws={"shrink": 0.8}, ax=ax,yticklabels=tick_gaps,xticklabels=tick_gaps)
        hm.invert_yaxis()
        ax.tick_params(labelsize=15)
        ax.set_title(different_residue_names[i])

    # Add a single x-axis label for the entire figure
    if data_type=='angle':
        fig.text(0.5, 0, name_latex[0]+f", degrees{additional_label}",fontsize=30, ha='center')
    elif data_type=='length':
        fig.text(0.5, 0, name_latex[0]+f", Angstroms{additional_label}",fontsize=30, ha='center')
    else:
        fig.text(0.5, 0, name_latex[0]+f", {additional_label}",fontsize=30, ha='center')

    if data_type=='angle':
        fig.text(0, 0.5, name_latex[1]+f", degrees{additional_label}",fontsize=30, va='center', rotation='vertical')
    elif data_type=='length':
        fig.text(0, 0.5, name_latex[1]+f", Angstroms{additional_label}",fontsize=30, va='center', rotation='vertical')
    else:
        fig.text(0, 0.5, name_latex[1]+f", {additional_label}",fontsize=30, va='center', rotation='vertical')

    plt.tight_layout()

    if log_scale:
        plt.savefig(output_folder+"PDB727K_{}_heatmap_{}_vs_{}_log_scale{}".format(invariant,name[0],name[1],additional)+".PNG",facecolor='white',transparent=False,
               bbox_inches="tight")
    else:
        plt.savefig(output_folder+"PDB727K_{}_heatmap_{}_vs_{}{}".format(invariant,name[0],name[1],additional)+".PNG",facecolor='white',transparent=False,
               bbox_inches="tight")
    plt.show()
    
    
    
def run_heatmap_grid(histograms,idx,output_folder,output_folder_log,invariant='AL',shift=False):


    #al pair plots
    if invariant=='AL':

        if shift:
            histograms_shifted = shift_ranges(histograms)
            for i in idx:
                xedges = histograms_shifted[1][int(20*i)]
                yedges = histograms_shifted[2][int(20*i)]
                xticks = [int(xedges.min()),int(xedges.max()),90]
                yticks = [int(yedges.min()),int(yedges.max()),90]
                name = al_names_2d[i]
                name_latex = al_names_2d_latex[i]
                data = [i.T for i in histograms_shifted[0][(20*i):(20*(i+1))]]
                factor=5
                tick_gaps=180
                data_type='angle'

                grid_heatmap_plot(data,xedges,yedges,tick_gaps,name,name_latex,output_folder,data_type,invariant,factor,angle=True,log_scale=False,additional="_range_shifted")
                grid_heatmap_plot(data,xedges,yedges,tick_gaps,name,name_latex,output_folder_log,data_type,invariant,factor,angle=True,log_scale=True,additional="_range_shifted")


        for i in idx:

            xedges = histograms[1][int(20*i)]
            yedges = histograms[2][int(20*i)]
            name = al_names_2d[i]
            name_latex = al_names_2d_latex[i]
            data = [i.T for i in histograms[0][(20*i):(20*(i+1))]]
            factor=5
            tick_gaps=10

            if i in [0,1,2]:
                data = [i[80:200,80:200] for i in data]
                xedges = xedges[80:201]
                yedges = yedges[80:201]
                tick_gaps=10
                data_type = 'length'
                factor=2
            elif i in [3,4,5]:
                tick_gaps=180
                data_type = 'angle'
            elif i in [6,7,8]:
                tick_gaps=180
                data_type = 'angle'

            grid_heatmap_plot(data,xedges,yedges,tick_gaps,name,name_latex,output_folder,data_type,invariant,factor,angle=True,log_scale=False,additional="")
            grid_heatmap_plot(data,xedges,yedges,tick_gaps,name,name_latex,output_folder_log,data_type,invariant,factor,angle=True,log_scale=True,additional="")


    if invariant=='BRI' or invariant=='TRIN':

        if invariant=='BRI':
            names_2d = bri_names_2d
            names_2d_latex = bri_names_2d_latex
            tick_gaps=10
        else:
            names_2d = trin_names_2d
            names_2d_latex = trin_names_2d_latex
            tick_gaps = 10

        for i in idx:

            xedges = histograms[1][int(20*i)]
            yedges = histograms[2][int(20*i)]
            name = names_2d[i]
            name_latex = names_2d_latex[i]
            data = [i.T for i in histograms[0][(20*i):(20*(i+1))]]
            data_type = 'length'
            factor=5


            grid_heatmap_plot(data,xedges,yedges,tick_gaps,name,name_latex,output_folder,data_type,invariant,factor,angle=True,log_scale=False,additional="_range_shifted")
            grid_heatmap_plot(data,xedges,yedges,tick_gaps,name,name_latex,output_folder_log,data_type,invariant,factor,angle=True,log_scale=True,additional="_range_shifted")
            
            
            
def create_histogram_plot(data,data_type,xedges,name,name_latex,output_folder,invariant,xticks,fs=30,log_scale=True,suffix="",additional=""):

    # Plot the histogram using plt.hist with pre-computed bins and counts
    fig, ax = plt.subplots()
    fig.set_figwidth(12)
    fig.set_figheight(8)

    sns.set(font_scale=2.5)
    sns.set_style("whitegrid")

    ax.hist(xedges[:-1], bins=xedges, weights=data, edgecolor='cornflowerblue',color='cornflowerblue')

    # Set labels and title
    if data_type=='angle':
        plt.xlabel(name_latex+", degrees{}".format(additional),fontsize=fs)
    elif data_type=='length':
        plt.xlabel(name_latex+", Angstroms{}".format(additional),fontsize=fs)
    else:
        plt.xlabel(name_latex+" {}".format(additional),fontsize=fs)

    plt.ylabel('Number of residues',fontsize=fs)

    if log_scale:
        plt.yscale('log')

    #min, max and spacing
    #x_ticks = np.arange(xticks[0],xticks[1]+xticks[2],xticks[2])

    #xedges_min = min(xedges)
    #xedges_gap = xedges[1]-xedges[0]
    #xticks_idx = np.arange(int(np.round((xticks[0]-xedges_min)/xedges_gap,1)),int(np.round((xticks[1]-xedges_min)/xedges_gap,1))+int(round(xticks[2]/xedges_gap,1)),int(round(xticks[2]/xedges_gap,1)))
    #plt.xticks(ticks=xticks_idx, labels=x_ticks,rotation=90)

    if log_scale:
        plt.savefig(output_folder+"PDB727K_{}_histogram_{}_log{}".format(invariant,name,suffix)+".PNG",facecolor='white', transparent=False,bbox_inches="tight")
    else:
        plt.savefig(output_folder+"PDB727K_{}_histogram_{}{}".format(invariant,name,suffix)+".PNG",facecolor='white', transparent=False,bbox_inches="tight")
    plt.show()
    
    
def plot_histograms(histograms,idx,output_folder,output_folder_log,invariant='AL',shift=False):

    #al pair plots
    if invariant=='AL':

        if shift:
            histograms_shifted = shift_ranges(histograms)
            for i in idx:
                xedges = histograms_shifted[4][int(20*i)]
                xticks = [int(xedges.min()),int(xedges.max()),90]
                name = al_names_1d[i]
                name_latex = al_names_1d_latex[i]
                data_type='angle'
                data = np.sum(histograms_shifted[3][(20*i):(20*(i+1))],axis=0)

                create_histogram_plot(data,data_type,xedges,name,name_latex,output_folder,invariant,xticks,fs=30,log_scale=False,suffix="_shifted_range",additional="")
                create_histogram_plot(data,data_type,xedges,name,name_latex,output_folder_log,invariant,xticks,fs=30,log_scale=True,suffix="_shifted_range",additional="")


        for i in idx:

            xedges = histograms[4][int(20*i)]
            name = al_names_1d[i]
            name_latex = al_names_1d_latex[i]
            data = np.sum(histograms[3][(20*i):(20*(i+1))],axis=0)

            if i in [0,1,2]:
                xticks = [0,3,0.5]
                data_type = 'length'
            elif i in [3,4,5]:
                xticks = [0,180,45]
                data_type = 'angle'
            elif i in [6,7,8]:
                xticks = [-180,180,90]
                data_type = 'angle'
                
            create_histogram_plot(data,data_type,xedges,name,name_latex,output_folder,invariant,xticks,fs=30,log_scale=False,suffix="",additional="")
            create_histogram_plot(data,data_type,xedges,name,name_latex,output_folder_log,invariant,xticks,fs=30,log_scale=True,suffix="",additional="")



    if invariant=='BRI' or invariant=='TRIN':

        if invariant=='BRI':
            names_1d = bri_names_1d
            names_1d_latex = bri_names_1d_latex
        else:
            names_1d = trin_names_1d
            names_1d_latex = trin_names_1d_latex

        for i in idx:

            xedges = histograms[4][int(20*i)]
            name = names_1d[i]
            name_latex = names_1d_latex[i]
            data = np.sum(histograms[3][(20*i):(20*(i+1))],axis=0)
            data_type = 'length'
            xticks = [-2,2,0.5]

            create_histogram_plot(data,data_type,xedges,name,name_latex,output_folder,invariant,xticks,fs=30,log_scale=False,suffix="",additional="")
            create_histogram_plot(data,data_type,xedges,name,name_latex,output_folder_log,invariant,xticks,fs=30,log_scale=True,suffix="",additional="")