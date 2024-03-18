import torch
import torch.nn as nn
import random
import time
from pypower.api import case14, case30, case118, case300, ppoption, runpf, printpf
import pickle as pkl
# from GNS_main import local_power_imbalance, global_power_imbalance, prepare_grid, LearningBlock, get_BLG

# purpose of this file is to evaluate the GNS model versus the DC algorithm, comparing the percentage errors of the active power generation/v and theta, to the ones from newton raphson, which is the most accurate

# load the grid
case_nr = 14  # 14, 30, 118, 300
augmentation_nr = random.randint(0, 9)
case_augmented =  pkl.load(open(f'./data/case{case_nr}/augmented_case{case_nr}_{augmentation_nr}.pkl', 'rb'))
# get optimal solution from newton raphson from library
newton_raphson = ppoption(PF_ALG=1)  # newton's method
start = time.perf_counter()
solved_grid_nr = runpf(case_augmented, newton_raphson)
stop = time.perf_counter()
duration_nr = stop - start
solved_grid_nr_gen = solved_grid_nr[0]['gen']

# DC algorithm
dc_solver = ppoption(PF_ALG=2)  # DC algorithm, XB version, neglecting influence on reactive power flows
start = time.perf_counter()
solved_grid_dc = runpf(case_augmented, dc_solver)
stop = time.perf_counter()
duration_dc = stop - start
solved_grid_dc_gen = solved_grid_dc['gen']

# GNS
# latent_dim = 10
# hidden_dim = 10
# run = 1620
# B, L, G = get_BLG()
# model = torch.load(f'models/best_model_c{case_nr}_K{K}_L{latent_dim}_H{hidden_dim}_I{run}.pth')
# model.eval()
# start = time.perf_counter()
# buses, lines, generators = prepare_grid(case_nr, augmentation_nr)
# v_gns, theta_gns, loss_gns = model(buses, lines, generators, B, L, G)
# stop = time.perf_counter()
# duration_gns = stop - start

# # some nice graphs and metrics
# v_error = torch.abs(1 - v_gns)
# theta_error = torch.abs(1 - theta_gns)
# time_diff = duration_nr - duration_gns





