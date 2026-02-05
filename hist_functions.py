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

def calculate_means(bri_al_data):
    # 1. Define columns
    cols_to_compute_mean_for = [
        'x(N)', 'y(N)', 'z(N)', 'x(A)', 'y(A)', 'z(A)', 'x(C)', 'y(C)', 'z(C)',
        'length(N)', 'length(A)', 'length(C)', 'angle(N)', 'angle(A)', 'angle(C)',
        'x(AN)', 'x(AC)', 'y(AC)'
    ]

    cols_to_compute_absolute_mean_for = ['tau(NA)', 'tau(AC)', 'tau(CN)']

    group_cols = ['pdb_id', 'model_id', 'chain_id', 'start_residue', 'chain_length']

    # 2. Create a copy to avoid Modifying the original dataframe in place
    # (Optional but recommended for safety)
    df_temp = bri_al_data.copy()

    # 3. Pre-calculate the absolute values
    # This transforms the 'tau' columns to |tau| instantly across the whole dataset
    df_temp[cols_to_compute_absolute_mean_for] = df_temp[cols_to_compute_absolute_mean_for].abs()

    # 4. Combine all target columns
    all_target_cols = cols_to_compute_mean_for + cols_to_compute_absolute_mean_for

    # 5. Perform a SINGLE GroupBy
    # Since we already took the absolute value of the tau columns,
    # .mean() will now correctly give us the "mean of the absolute values"
    result_df = df_temp.groupby(group_cols)[all_target_cols].mean().reset_index()

    return result_df
    
    
pickled_histograms_directory = "./data/pickled_histograms/"
if not os.path.exists(pickled_histograms_directory):
  os.makedirs(pickled_histograms_directory)

def pickler_write(name,obj,restrict=False):
    directory = pickled_histograms_directory
    #print(f'pickling {name}')
    if restrict:
        with open(directory+"/"+name+"_restrict.pkl", 'wb') as f:
            pickle.dump(obj, f)
    else:
        with open(directory+"/"+name+".pkl", 'wb') as f:
            pickle.dump(obj, f)

def pickler_read(name,restrict=False):
    directory = pickled_histograms_directory
    #print(f'reading pickle {name}')
    if restrict:
        with open(directory+"/"+name+"_restrict.pkl", 'rb') as f:
            obj = pickle.load(f)
    else:
        with open(directory+"/"+name+".pkl", 'rb') as f:
            obj = pickle.load(f)
    return(obj)
    
def create_histogram2d(results, pairs, input_params):
    output_histograms = []
    output_xedges = []
    output_yedges = []

    different_residue_names = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

    for i in range(len(pairs)):
        pair = pairs[i]
        ip = input_params[i]
        x_bin_size, y_bin_size, x_min, x_max, y_min, y_max = ip[0], ip[1], ip[2], ip[3], ip[4], ip[5]

        # Select data for this pair
        data = results[['residue_label', pair[0], pair[1]]].copy()
        data = data[~(data[pair[0]].isna()) & ~(data[pair[1]].isna())]

        # Define bins
        x_bins = np.arange(x_min, x_max + x_bin_size, x_bin_size)
        y_bins = np.arange(y_min, y_max + y_bin_size, y_bin_size)

        # --- OPTIMIZATION: Group by residue once instead of filtering inside the loop ---
        grouped_data = data.groupby('residue_label')

        for res in different_residue_names:
            if res in grouped_data.groups:
                # Get the group directly
                results_res = grouped_data.get_group(res).to_numpy()
                x = results_res[:, 1].astype(float)
                y = results_res[:, 2].astype(float)
            else:
                # Handle missing residues gracefully
                x = np.array([])
                y = np.array([])

            # Create 2D histogram
            heatmap, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins], range=[[x_min, x_max], [y_min, y_max]])

            # Compress to sparse matrix
            heatmap = sp.csr_matrix(heatmap)

            output_histograms.append(heatmap)
            output_xedges.append(xedges)
            output_yedges.append(yedges)

    return output_histograms, output_xedges, output_yedges


def create_histogram(results, columns, input_params):
    output_histograms = []
    output_xedges = []

    different_residue_names = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

    for i in range(len(columns)):
        col = columns[i]
        ip = input_params[i]
        x_bin_size, x_min, x_max = ip[0], ip[1], ip[2]

        data = results[['residue_label', col]].copy()
        data = data[~(data[col].isna())]

        # Define bins
        x_bins = np.arange(x_min, x_max + x_bin_size, x_bin_size)

        # --- OPTIMIZATION: Group by residue once ---
        grouped_data = data.groupby('residue_label')

        for res in different_residue_names:
            if res in grouped_data.groups:
                results_res = grouped_data.get_group(res).to_numpy()
                x = results_res[:, 1].astype(float)
            else:
                x = np.array([])

            heatmap, xedges = np.histogram(x, bins=x_bins, range=[x_min, x_max])

            output_histograms.append(heatmap)
            output_xedges.append(xedges)

    return output_histograms, output_xedges


def create_histogram2d_means(results, pairs, input_params):
    output_histograms = []
    output_xedges = []
    output_yedges = []

    for i in range(len(pairs)):
        pair = pairs[i]
        ip = input_params[i]
        x_bin_size, y_bin_size, x_min, x_max, y_min, y_max = ip[0], ip[1], ip[2], ip[3], ip[4], ip[5]

        data = results[[pair[0], pair[1]]].copy()
        data = data[~(data[pair[0]].isna()) & ~(data[pair[1]].isna())]

        x_bins = np.arange(x_min, x_max + x_bin_size, x_bin_size)
        y_bins = np.arange(y_min, y_max + y_bin_size, y_bin_size)

        results_res = data.to_numpy()
        x = results_res[:, 0].astype(float)
        y = results_res[:, 1].astype(float)

        # 1. Calculate the actual mean histogram
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins], range=[[x_min, x_max], [y_min, y_max]])
        heatmap = sp.csr_matrix(heatmap)

        # Append the actual data (Entry 1)
        output_histograms.append(heatmap)
        output_xedges.append(xedges)
        output_yedges.append(yedges)

        # 2. Pad with 19 empty matrices to match the 20-amino-acid structure
        # Create an empty sparse matrix of the exact same shape
        empty_heatmap = sp.csr_matrix(heatmap.shape, dtype=heatmap.dtype)

        for _ in range(19):
            output_histograms.append(empty_heatmap)
            output_xedges.append(xedges) # We repeat the edges so indices stay aligned
            output_yedges.append(yedges)

    return output_histograms, output_xedges, output_yedges


def create_histogram_means(results, columns, input_params):
    output_histograms = []
    output_xedges = []

    for i in range(len(columns)):
        col = columns[i]
        ip = input_params[i]
        x_bin_size, x_min, x_max = ip[0], ip[1], ip[2]

        data = results[[col]].copy()
        data = data[~(data[col].isna())]

        results_res = data.to_numpy()
        x = results_res[:, 0].astype(float)

        x_bins = np.arange(x_min, x_max + x_bin_size, x_bin_size)
        
        # 1. Calculate actual mean histogram
        heatmap, xedges = np.histogram(x, bins=x_bins, range=[x_min, x_max])

        # Append the actual data (Entry 1)
        output_histograms.append(heatmap)
        output_xedges.append(xedges)

        # 2. Pad with 19 empty arrays
        empty_heatmap = np.zeros_like(heatmap)

        for _ in range(19):
            output_histograms.append(empty_heatmap)
            output_xedges.append(xedges)

    return output_histograms, output_xedges
    
# @title Read and combine pickles

def read_data(unique_first_two, should_restrict, mean=False):

    index = 0

    # Determine the key suffix to look for inside the dictionary
    # If mean=False, we look for 'al_2d'. If mean=True, we look for 'al_2d_mean'
    suffix = "_mean" if mean else ""

    for i in tqdm.tqdm(unique_first_two):
        try:
            # Load the single combined dictionary for this file
            combined_data = pickler_read(f"{i}_data", restrict=should_restrict)

            # Extract the specific datasets we need based on the suffix
            # Each retrieve returns the tuple (histograms, xedges, yedges)
            al_2d_data = combined_data.get(f"al_2d{suffix}")
            bri_2d_data = combined_data.get(f"bri_2d{suffix}")
            trin_2d_data = combined_data.get(f"trin_2d{suffix}")

            al_1d_data = combined_data.get(f"al_1d{suffix}")
            bri_1d_data = combined_data.get(f"bri_1d{suffix}")
            trin_1d_data = combined_data.get(f"trin_1d{suffix}")

            # Check if we successfully got data (in case some files lacked means/standard data)
            if None in [al_2d_data, bri_2d_data, trin_2d_data]:
                continue

            # Unpack the histograms from the tuple (hist, x, y)
            output_histograms_al_2d = al_2d_data[0]
            output_histograms_bri_2d = bri_2d_data[0]
            output_histograms_trin_2d = trin_2d_data[0]

            output_histograms_al_1d = al_1d_data[0]
            output_histograms_bri_1d = bri_1d_data[0]
            output_histograms_trin_1d = trin_1d_data[0]

        except Exception as e:
            continue

        # --- Aggregate Histograms ---
        if index == 0:
            overall_output_histograms_al_2d = output_histograms_al_2d.copy()
            overall_output_histograms_bri_2d = output_histograms_bri_2d.copy()
            overall_output_histograms_trin_2d = output_histograms_trin_2d.copy()

            overall_output_histograms_al_1d = output_histograms_al_1d.copy()
            overall_output_histograms_bri_1d = output_histograms_bri_1d.copy()
            overall_output_histograms_trin_1d = output_histograms_trin_1d.copy()

            # Store edges from the first successful file
            # 2D edges
            overall_output_xedges_al_2d = al_2d_data[1]
            overall_output_yedges_al_2d = al_2d_data[2]
            overall_output_xedges_bri_2d = bri_2d_data[1]
            overall_output_yedges_bri_2d = bri_2d_data[2]
            overall_output_xedges_trin_2d = trin_2d_data[1]
            overall_output_yedges_trin_2d = trin_2d_data[2]

            # 1D edges
            overall_output_xedges_al_1d = al_1d_data[1]
            overall_output_xedges_bri_1d = bri_1d_data[1]
            overall_output_xedges_trin_1d = trin_1d_data[1]
        else:
            # Sum the lists of sparse matrices
            overall_output_histograms_al_2d = [sum(x) for x in zip(overall_output_histograms_al_2d, output_histograms_al_2d)]
            overall_output_histograms_bri_2d = [sum(x) for x in zip(overall_output_histograms_bri_2d, output_histograms_bri_2d)]
            overall_output_histograms_trin_2d = [sum(x) for x in zip(overall_output_histograms_trin_2d, output_histograms_trin_2d)]

            overall_output_histograms_al_1d = [sum(x) for x in zip(overall_output_histograms_al_1d, output_histograms_al_1d)]
            overall_output_histograms_bri_1d = [sum(x) for x in zip(overall_output_histograms_bri_1d, output_histograms_bri_1d)]
            overall_output_histograms_trin_1d = [sum(x) for x in zip(overall_output_histograms_trin_1d, output_histograms_trin_1d)]

        index += 1

    # Package results for return
    al = [overall_output_histograms_al_2d, overall_output_xedges_al_2d, overall_output_yedges_al_2d,
          overall_output_histograms_al_1d, overall_output_xedges_al_1d]

    bri = [overall_output_histograms_bri_2d, overall_output_xedges_bri_2d, overall_output_yedges_bri_2d,
          overall_output_histograms_bri_1d, overall_output_xedges_bri_1d]

    trin = [overall_output_histograms_trin_2d, overall_output_xedges_trin_2d, overall_output_yedges_trin_2d,
          overall_output_histograms_trin_1d, overall_output_xedges_trin_1d]

    return(al, bri, trin)
    
