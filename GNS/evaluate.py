import torch
import torch.nn as nn
import random
import time
from pypower.api import case14, case30, case118, case300, ppoption, runpf, printpf
import pickle as pkl
from utils import load_all_grids, prepare_grid, get_BLG
import matplotlib.pyplot as plt
# from GNS_main import local_power_imbalance, global_power_imbalance, prepare_grid, LearningBlock, get_BLG


# purpose of this file is to evaluate the GNS model versus the DC algorithm, comparing the percentage errors of the active power generation/v and theta, to the ones from newton raphson, which is the most accurate

# load the grid
case_nr = 14  # 14, 30, 118, 300
nr_eval_samples = 100

# get optimal solution from newton raphson from library
newton_raphson = ppoption(PF_ALG=1)  # newton's method
NR_duration_times = torch.zeros(nr_eval_samples, dtype=torch.float32)
NR_theta_out = torch.zeros((nr_eval_samples, case_nr), dtype=torch.float32)
NR_v_out = torch.zeros((nr_eval_samples, case_nr), dtype=torch.float32)
for i, grid_i in enumerate(range(1000-nr_eval_samples, 1001)):
    case_augmented = pkl.load(open(f'../data/case{case_nr}/augmented_case{case_nr}_{grid_i}.pkl', 'rb'))
    start = time.perf_counter()
    solved_grid_nr = runpf(case_augmented, newton_raphson)
    stop = time.perf_counter()
    duration_nr = stop - start
    NR_duration_times[i] = duration_nr
    NR_theta_out[i] = solved_grid_nr[0]['bus'][:, 8]
    NR_v_out[i] = solved_grid_nr[0]['bus'][:, 7]


# DC algorithm
dc_solver = ppoption(PF_ALG=2)  # DC algorithm, XB version, neglecting influence on reactive power flows
DC_duration_times = torch.zeros(nr_eval_samples, dtype=torch.float32)
DC_theta_out = torch.zeros((nr_eval_samples, case_nr), dtype=torch.float32)
DC_v_out = torch.zeros((nr_eval_samples, case_nr), dtype=torch.float32)
for i, grid_i in enumerate(range(1000-nr_eval_samples, 1001)):
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
model = torch.load(f'models/best_model_c{case_nr}_K{K}_L{latent_dim}_H{hidden_dim}_B{batch_size}.pth')
B, L, G = get_BLG()
model.eval()
GNS_duration_times = torch.zeros(nr_eval_samples, dtype=torch.float32)
GNS_theta_out = torch.zeros((nr_eval_samples, case_nr), dtype=torch.float32)
GNS_v_out = torch.zeros((nr_eval_samples, case_nr), dtype=torch.float32)
for i, grid_i in enumerate(range(1000-nr_eval_samples, 1001)):
    case_augmented = pkl.load(open(f'../data/case{case_nr}/augmented_case{case_nr}_{grid_i}.pkl', 'rb'))
    buses, lines, generators = prepare_grid(case_nr, i)
    start = time.perf_counter()
    v_gns, theta_gns, loss_gns = model(buses, lines, generators, B, L, G)
    stop = time.perf_counter()
    duration_gns = stop - start
    GNS_duration_times[i] = duration_gns
    GNS_theta_out[i] = theta_gns
    GNS_v_out[i] = v_gns


# some nice graphs and metrics comparing v and theta of the three methods
time_diff_gns_nr = GNS_duration_times - NR_duration_times
time_diff_nr_dc = NR_duration_times - DC_duration_times
mean_diff_time = torch.mean(time_diff_gns_nr)
std_diff_time = torch.std(time_diff_gns_nr)

theta_diff_gns_nr = torch.abs(GNS_theta_out - NR_theta_out)
theta_diff_nr_dc = torch.abs(NR_theta_out - DC_theta_out)
mean_diff_theta = torch.mean(theta_diff_gns_nr)
std_diff_theta = torch.std(theta_diff_gns_nr)

v_diff_gns_nr = torch.abs(GNS_v_out - NR_v_out)
v_diff_nr_dc = torch.abs(NR_v_out - DC_v_out)
mean_diff_v = torch.mean(v_diff_gns_nr)
std_diff_v = torch.std(v_diff_gns_nr)

#plotting on logarithmic scale
plt.figure()
plt.plot(torch.arange(nr_eval_samples), time_diff_gns_nr, label='GNS - NR')
plt.plot(torch.arange(nr_eval_samples), time_diff_nr_dc, label='NR - DC')
plt.yscale('log')
plt.legend()
# mean and std overview table print
print(f'Mean time difference GNS - NR: {mean_diff_time}, std: {std_diff_time}')
print(f'Mean time difference NR - DC: {mean_diff_time}, std: {std_diff_time}')





