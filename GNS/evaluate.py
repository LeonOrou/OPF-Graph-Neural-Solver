import torch
import torch.nn as nn
import random
import time
from pypower.api import case14, case30, case118, case300, ppoption, runpf, printpf
import pickle as pkl
from utils import load_all_grids, prepare_grid, get_BLG
# import matplotlib as plt
import numpy as np
from GCN_main import GNS
# from GNS_main import local_power_imbalance, global_power_imbalance, prepare_grid, LearningBlock, get_BLG


# purpose of this file is to evaluate the GNS model versus the DC algorithm, comparing the percentage errors of the active power generation/v and theta, to the ones from newton raphson, which is the most accurate

# load the grid
case_nr = 14  # 14, 30, 118, 300
nr_eval_samples = 100

# get optimal solution from newton raphson from library
newton_raphson = ppoption(PF_ALG=1, verbose=0)  # newton's method
NR_duration_times = np.zeros(nr_eval_samples, dtype=np.float32)
NR_theta_out = np.zeros((nr_eval_samples, case_nr), dtype=np.float32)
NR_v_out = np.zeros((nr_eval_samples, case_nr), dtype=np.float32)
for i, grid_i in enumerate(range(1001-nr_eval_samples, 1001)):
    case_augmented = pkl.load(open(f'../data/case{case_nr}/augmented_case{case_nr}_{grid_i}.pkl', 'rb'))
    start = time.perf_counter()
    solved_grid_nr = runpf(case_augmented, newton_raphson)
    stop = time.perf_counter()
    duration_nr = stop - start
    NR_duration_times[i] = duration_nr
    NR_theta_out[i] = solved_grid_nr[0]['bus'][:, 8]
    NR_v_out[i] = solved_grid_nr[0]['bus'][:, 7]


# DC algorithm
dc_solver = ppoption(PF_ALG=2, verbose=0)  # DC algorithm, XB version, neglecting influence on reactive power flows
DC_duration_times = np.zeros(nr_eval_samples, dtype=np.float32)
DC_theta_out = np.zeros((nr_eval_samples, case_nr), dtype=np.float32)
DC_v_out = np.zeros((nr_eval_samples, case_nr), dtype=np.float32)
for i, grid_i in enumerate(range(1001-nr_eval_samples, 1001)):
    case_augmented = pkl.load(open(f'../data/case{case_nr}/augmented_case{case_nr}_{grid_i}.pkl', 'rb'))
    start = time.perf_counter()
    solved_grid_dc = runpf(case_augmented, dc_solver)
    stop = time.perf_counter()
    duration_dc = stop - start
    DC_duration_times[i] = duration_dc
    DC_theta_out[i] = solved_grid_dc[0]['bus'][:, 8]
    DC_v_out[i] = solved_grid_dc[0]['bus'][:, 7]

# GNS
case_nr = 14
K = 10
latent_dim = 10
hidden_dim = 10
batch_size = 5
model = GNS(latent_dim=latent_dim, hidden_dim=hidden_dim, K=K)
model.load_state_dict(torch.load(f'../models/best_model_c14_K10_L10_H10.pth'))
B, L, G = get_BLG()
GNS_duration_times = np.zeros(nr_eval_samples, dtype=np.float32)
GNS_theta_out = np.zeros((nr_eval_samples, case_nr), dtype=np.float32)
GNS_v_out = np.zeros((nr_eval_samples, case_nr), dtype=np.float32)
for i, grid_i in enumerate(range(1001-nr_eval_samples, 1001)):
    case_augmented = pkl.load(open(f'../data/case{case_nr}/augmented_case{case_nr}_{grid_i}.pkl', 'rb'))
    buses, lines, generators = prepare_grid(case_nr, i)
    start = time.perf_counter()
    v_gns, theta_gns, loss_gns = model(buses, lines, generators, B, L, G)
    stop = time.perf_counter()
    duration_gns = stop - start
    GNS_duration_times[i] = duration_gns
    GNS_theta_out[i] = theta_gns.detach().numpy()
    GNS_v_out[i] = v_gns.detach().numpy()


# some nice graphs and metrics comparing v and theta of the three methods
time_diff_gns_nr = np.abs(GNS_duration_times - NR_duration_times)
time_diff_nr_dc = np.abs(NR_duration_times - DC_duration_times)
mean_diff_time_gns = np.mean(time_diff_gns_nr)
std_diff_time_gns = np.std(time_diff_gns_nr)
mean_diff_time_nr = np.mean(time_diff_nr_dc)
std_diff_time_nr = np.std(time_diff_nr_dc)

theta_diff_gns_nr = np.abs(GNS_theta_out - NR_theta_out)
theta_diff_nr_dc = np.abs(NR_theta_out - DC_theta_out)
mean_diff_theta_gns = np.mean(theta_diff_gns_nr)
std_diff_theta_gns = np.std(theta_diff_gns_nr)
mean_diff_theta_nr = np.mean(theta_diff_nr_dc)
std_diff_theta_nr = np.std(theta_diff_nr_dc)

v_diff_gns_nr = np.abs(GNS_v_out - NR_v_out)
v_diff_nr_dc = np.abs(NR_v_out - DC_v_out)
mean_diff_v_gns = np.mean(v_diff_gns_nr)
std_diff_v_gns = np.std(v_diff_gns_nr)
mean_diff_v_nr = np.mean(v_diff_nr_dc)
std_diff_v_nr = np.std(v_diff_nr_dc)

#plotting on logarithmic scale
# plt.pyplot.figure()
# plt.pyplot.plot(np.arange(nr_eval_samples), time_diff_gns_nr, label='GNS - NR')
# plt.pyplot.plot(np.arange(nr_eval_samples), time_diff_nr_dc, label='NR - DC')
# plt.yscale('log')
# plt.legend()
# mean and std overview table print for time, v and theta
print(f'Time difference GNS and NR: Mean: {np.round(mean_diff_time_gns, 5)}, Std: {np.round(std_diff_time_gns, 5)}')
print(f'Time difference NR and DC: Mean: {np.round(mean_diff_time_nr, 5)}, Std: {np.round(std_diff_time_nr, 5)}')
print(f'Theta difference GNS and NR: Mean: {np.round(mean_diff_theta_gns, 5)}, Std: {np.round(std_diff_theta_gns, 5)}')
print(f'Theta difference NR and DC: Mean: {np.round(mean_diff_theta_nr, 5)}, Std: {np.round(std_diff_theta_nr, 5)}')
print(f'V difference GNS and NR: Mean: {np.round(mean_diff_v_gns, 5)}, Std: {np.round(std_diff_v_gns, 5)}')
print(f'V difference NR and DC: Mean: {np.round(mean_diff_v_nr, 5)}, Std: {np.round(std_diff_v_nr, 5)}')






