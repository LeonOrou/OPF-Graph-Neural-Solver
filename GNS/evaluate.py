import torch
import time
from pypower.api import case9, case14, case30, case118, case300, ppoption, runpf, printpf
from pypower import dcopf_solver, dcopf
import pickle as pkl
from utils import load_all_grids, prepare_grid, get_BLG
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from main import GNS
# from GNS_main import local_power_imbalance, global_power_imbalance, prepare_grid, LearningBlock, get_BLG

# purpose of this file is to evaluate the GNS model versus the DC algorithm, comparing the percentage errors of the active power generation/v and theta, to the ones from newton raphson, which is the most accurate

def active_line_flow(V, theta, x, src, dst):
    src = src.astype(int)-1  # -1 as bus nr are natural numbers
    dst = dst.astype(int)-1
    return 1/x*(V[src]*V[dst]*np.sin(theta[src]-theta[dst]))

# load the grid
case_nr = 14  # 14, 30, 118, 300
nr_eval_samples = 1024

# get optimal solution from newton raphson from library
newton_raphson = ppoption(PF_ALG=1)  # newton's method
newton_raphson = ppoption(newton_raphson, VERBOSE=0)
NR_duration_times = np.zeros(nr_eval_samples, dtype=np.float32)
NR_theta_out = np.zeros((nr_eval_samples, case_nr), dtype=np.float32)
NR_v_out = np.zeros((nr_eval_samples, case_nr), dtype=np.float32)
NR_active_line_flow = np.zeros((nr_eval_samples, 20), dtype=np.float32)
for i, grid_i in enumerate(range(10001-nr_eval_samples, 10001)):
    case_augmented = pkl.load(open(f'../data/case{case_nr}/augmented_case{case_nr}_{grid_i}.pkl', 'rb'))
    start = time.perf_counter()
    solved_grid_nr = runpf(case_augmented, newton_raphson)
    stop = time.perf_counter()
    duration_nr = stop - start
    NR_duration_times[i] = duration_nr
    NR_theta_out[i] = solved_grid_nr[0]['bus'][:, 8]
    NR_v_out[i] = solved_grid_nr[0]['bus'][:, 7]
    NR_active_line_flow[i] = active_line_flow(solved_grid_nr[0]['bus'][:, 7], solved_grid_nr[0]['bus'][:, 8], solved_grid_nr[0]['branch'][:, 3], solved_grid_nr[0]['branch'][:, 0], solved_grid_nr[0]['branch'][:, 1])

# DC algorithm
# dc_solver = dcopf_solver()
# dc_solver = dcopf_solver(dc_solver, VERBOSE=0)

# dc_solver = ppoption(PF_ALG=3)
# DC_duration_times = np.zeros(nr_eval_samples, dtype=np.float32)
# DC_theta_out = np.zeros((nr_eval_samples, case_nr), dtype=np.float32)
# DC_v_out = np.zeros((nr_eval_samples, case_nr), dtype=np.float32)
# for i, grid_i in enumerate(range(10001-nr_eval_samples, 10001)):
#     case_augmented = pkl.load(open(f'../data/case{case_nr}/augmented_case{case_nr}_{grid_i}.pkl', 'rb'))
#     start = time.perf_counter()
#     solved_grid_dc = runpf(case_augmented, dc_solver)
#     stop = time.perf_counter()
#     duration_dc = stop - start
#     DC_duration_times[i] = duration_dc
#     DC_theta_out[i] = solved_grid_dc[0]['bus'][:, 8]
#     DC_v_out[i] = solved_grid_dc[0]['bus'][:, 7]

# GNS
K = 6
latent_dim = 20
multiple_phi = False
torch.manual_seed(42)
model = GNS(latent_dim=latent_dim, K=K, multiple_phi=multiple_phi)
model.load_state_dict(torch.load(f'../models/Finished Models/best_model_c14_K{K}_L{latent_dim}_H10_{multiple_phi}_optimAdam.pth'))
# model.load_state_dict(torch.load(f'../models/Finished long term models/best_model_E625_c14_K6_L10_H10_True_optimAdam.pth'))
B, L, G = get_BLG()
GNS_duration_times = np.zeros(nr_eval_samples, dtype=np.float32)
GNS_theta_out = np.zeros((nr_eval_samples, case_nr), dtype=np.float32)
GNS_v_out = np.zeros((nr_eval_samples, case_nr), dtype=np.float32)
last_losses = np.zeros(nr_eval_samples, dtype=np.float32)
GNS_active_line_flow = np.zeros((nr_eval_samples, 20), dtype=np.float32)
for i, grid_i in enumerate(range(10001-nr_eval_samples, 10001)):
    case_augmented = pkl.load(open(f'../data/case{case_nr}/augmented_case{case_nr}_{grid_i}.pkl', 'rb'))
    buses, lines, generators = prepare_grid(case_nr, i)
    # making theta_shift rad is done in grid preperation inside utils.py
    start = time.perf_counter()
    v_gns, theta_gns, loss_gns, last_loss = model(buses, lines, generators, B, L, G)
    stop = time.perf_counter()
    duration_gns = stop - start
    GNS_duration_times[i] = duration_gns
    GNS_theta_out[i] = theta_gns.detach().numpy()
    GNS_v_out[i] = v_gns.detach().numpy()
    last_losses[i] = last_loss
    GNS_active_line_flow[i] = active_line_flow(v_gns.detach().numpy(), theta_gns.detach().numpy(), lines[:, 3].detach().numpy(), lines[:, 0].detach().numpy(), lines[:, 1].detach().numpy())

# some nice graphs and metrics comparing v and theta of the three methods
time_diff_gns_nr = GNS_duration_times - NR_duration_times
# time_diff_nr_dc = DC_duration_times - NR_duration_times
mean_diff_time_gns = np.mean(time_diff_gns_nr)
std_diff_time_gns = np.std(time_diff_gns_nr)
# mean_diff_time_nr = np.mean(time_diff_nr_dc)
# std_diff_time_nr = np.std(time_diff_nr_dc)

# convert theta to radians
# [1:] mabye because first one is always 0 somehow
NR_theta_out = np.deg2rad(NR_theta_out[:, 0:])  # [1:] because the first one is not solved in the algorithms
# DC_theta_out = np.deg2rad(DC_theta_out[:, 0:])
GNS_theta_out = GNS_theta_out[:, 0:]
theta_diff_gns_nr = np.abs(GNS_theta_out - NR_theta_out)
# theta_diff_nr_dc = np.abs(NR_theta_out - DC_theta_out)
mean_diff_theta_gns = np.mean(theta_diff_gns_nr)
std_diff_theta_gns = np.std(theta_diff_gns_nr)
# mean_diff_theta_nr = np.mean(theta_diff_nr_dc)
# std_diff_theta_nr = np.std(theta_diff_nr_dc)

v_diff_gns_nr = np.abs(GNS_v_out - NR_v_out)
# v_diff_nr_dc = np.abs(NR_v_out - DC_v_out)
mean_diff_v_gns = np.mean(v_diff_gns_nr)
std_diff_v_gns = np.std(v_diff_gns_nr)
# mean_diff_v_nr = np.mean(v_diff_nr_dc)
# std_diff_v_nr = np.std(v_diff_nr_dc)

# get percentual errors how much the results diverge from NR solution
theta_error_gns_nr = np.abs((GNS_theta_out - NR_theta_out) / NR_theta_out) * 100
# theta_error_nr_dc = np.abs((DC_theta_out - NR_theta_out) / NR_theta_out) * 100
v_error_gns_nr = np.abs((GNS_v_out - NR_v_out) / NR_v_out) * 100
# v_error_nr_dc = np.abs((DC_v_out - NR_v_out) / NR_v_out) * 100

alf_diff_gns_nr = NR_active_line_flow - GNS_active_line_flow
# mean_diff_alf_gns_nr = np.mean(alf_diff_gns_nr)
# std_diff_alf_gns_nr = np.std(alf_diff_gns_nr)
percentage_diff_alf_gns_nr = np.abs(alf_diff_gns_nr / NR_active_line_flow) * 100
# take only lowest 50% of the percentage differences
percentage_diff_alf_gns_nr = np.sort(percentage_diff_alf_gns_nr, axis=None)[:int(percentage_diff_alf_gns_nr.size/2)]
twenty_percentile_diff_alf_gns_nr = np.percentile(percentage_diff_alf_gns_nr, 20)
median_diff_alf_gns_nr = np.median(percentage_diff_alf_gns_nr)
eighty_percentile_diff_alf_gns_nr = np.percentile(percentage_diff_alf_gns_nr, 80)

#plotting on logarithmic scale
# plt.pyplot.figure()
# plt.pyplot.plot(np.arange(nr_eval_samples), time_diff_gns_nr, label='GNS - NR')
# plt.pyplot.plot(np.arange(nr_eval_samples), time_diff_nr_dc, label='NR - DC')
# plt.yscale('log')
# plt.legend()
# mean and std overview table print for time, v and theta
print(f'Time difference GNS and NR: Mean: {np.round(mean_diff_time_gns, 5)}, Std: {np.round(std_diff_time_gns, 5)}')
# print(f'Time difference NR and DC: Mean: {np.round(mean_diff_time_nr, 5)}, Std: {np.round(std_diff_time_nr, 5)}')
print(f'Theta difference GNS and NR: Mean: {np.round(mean_diff_theta_gns, 5)}, Std: {np.round(std_diff_theta_gns, 5)}')
# print(f'Theta difference NR and DC: Mean: {np.round(mean_diff_theta_nr, 5)}, Std: {np.round(std_diff_theta_nr, 5)}')
print(f'V difference GNS and NR: Mean: {np.round(mean_diff_v_gns, 5)}, Std: {np.round(std_diff_v_gns, 5)}')
# print(f'V difference NR and DC: Mean: {np.round(mean_diff_v_nr, 5)}, Std: {np.round(std_diff_v_nr, 5)}')
# print(f'Theta % error GNS and NR: Mean: {np.round(np.mean(theta_error_gns_nr), 5)}, Std: {np.round(np.std(theta_error_gns_nr), 5)}')
# print(f'Theta % error NR and DC: Mean: {np.round(np.mean(theta_error_nr_dc), 5)}, Std: {np.round(np.std(theta_error_nr_dc), 5)}')
print(f'GNS last loss: Mean: {np.round(np.mean(last_losses), 5)}, Std: {np.round(np.std(last_losses), 5)}')
# print(f'Active line flow difference GNS and NR: Mean: {np.round(mean_diff_alf_gns_nr, 5)}, Std: {np.round(std_diff_alf_gns_nr, 5)}')
print(f'Active line flow percentage difference GNS and NR: 20th percentile: {np.round(twenty_percentile_diff_alf_gns_nr, 5)}, Median: {np.round(median_diff_alf_gns_nr, 5)}, 80th percentile: {np.round(eighty_percentile_diff_alf_gns_nr, 5)}')
# correlation plots and saving the files automatically
# plot mean and std difference from NR result for every node (second dimension)
v_diff_gns_nr = NR_v_out - GNS_v_out
theta_diffs_gns_nr = NR_theta_out - GNS_theta_out

mean_diffs_v = np.mean(v_diff_gns_nr, axis=0)
std_diffs_v = np.std(v_diff_gns_nr, axis=0)
mean_diffs_theta = np.mean(theta_diff_gns_nr, axis=0)
std_diffs_theta = np.std(theta_diff_gns_nr, axis=0)

# plot the mean and std difference from NR result for every node
fig, ax1 = plt.subplots()
color1 = 'tab:blue'
ax1.errorbar(np.arange(1, case_nr+1), mean_diffs_v, std_diffs_v, color=color1, ecolor=color1, marker='o', linestyle='None', label='V', capsize=5, capthick=1)
ax1.set_xticks(np.arange(1, case_nr+1))
ax1.set_xlabel('Bus number')
ax1.set_ylabel('Error of GNS compared to NR')
# ax1.tick_params(axis='y', colors=color1)

# ax2 = ax1.twinx()
color2 = 'tab:orange'
ax1.errorbar(np.arange(1, case_nr+1), mean_diffs_theta, std_diffs_theta, color=color2, ecolor=color2, marker='o', linestyle='None', label='theta', capsize=5, capthick=1)
# ax2.set_ylabel('theta difference between GNS and NR', color=color2)
# ax2.tick_params(axis='y', colors=color2)

plt.title(f'V and Theta error with K={K}, L={latent_dim}, Distinct Phi={multiple_phi}')
fig.legend()
plt.grid(True)
plt.show()
plt.savefig(f'../images/mean_sdt_nodes_c14_K{K}_L{latent_dim}_{multiple_phi}.png')

# fig, ax = plt.subplots()
# for v_gns, v_nr in zip(GNS_v_out, NR_v_out):
#     y = v_nr - v_gns
#     x = np.arange(len(y))
#
#     xy = np.vstack([x, y])
#     z = gaussian_kde(xy)(xy)
#
#     scatter = ax.scatter(x, y, c=z, s=5, cmap='viridis')
#
# # Add color bar
# cbar = plt.colorbar(scatter)
# cbar.set_label('Density')
#
# # Calculate and display the correlation coefficient
# correlation = np.corrcoef(x, y)[0, 1]
# plt.text(0.05, 0.95, f'Cor = {correlation:.3f}', transform=ax.transAxes, fontsize=14, verticalalignment='top')
#
# # Set labels
# ax.set_xlabel('X-axis')
# ax.set_ylabel('DC (MW)')
#
# plt.show()
#
