import torch
# from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# from scipy.io import loadmat
# from torch_geometric.nn import MessagePassing
# from torch_geometric.data import Data
# from torch_geometric.utils import add_self_loops, degree
# from scipy.optimize import minimize
# import torch_geometric.transforms as T
# from torch_geometric.nn import SAGEConv, to_hetero
# import torch.nn.functional as F
import time

from pypower.api import case14, case30, case118, case300, ppoption, runpf, printpf
import numpy as np
import pickle as pkl
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

# paper for this whole project, very careful reading: https://pscc-central.epfl.ch/repo/papers/2020/715.pdf
# documentation of data formats for pypower, very careful reading: https://rwl.github.io/PYPOWER/api/pypower.caseformat-module.html

pf_net_pyp = case14()
bus_data = pf_net_pyp['bus']
branch_data = pf_net_pyp['branch']
gen_data = pf_net_pyp['gen']
cost_data = pf_net_pyp['gencost']

# ppopt = ppoption(PF_ALG=1)
# r = runpf(pf_net_pyp, ppopt)
# printpf(r)

# num_nodes = bus_data.shape[0]
# num_edges = branch_data.shape[0]

## case format: https://rwl.github.io/PYPOWER/api/pypower.caseformat-module.html


# add column qg to buses with default value 0 as we later use qg for each bus
B = {'bus_i': 0, 'type': 1, 'Pd': 2, 'Qd': 3, 'Gs': 4, 'Bs': 5, 'qg':6}  # indices of bus data
# baseKV = buses[:, 8]
# baseMV = baseKV / 1000

# lines = torch.tensor(branch_data[:, [0, 1, 2, 3, 4, 8, 9]], dtype=torch.float32)
L = {'f_bus': 0, 't_bus': 1, 'r': 2, 'x': 3, 'b': 4, 'tau': 5, 'theta': 6}  # indices of branch data
# if tau=0, set it to 1: 0 is default in pypower, but matpower default is 1 (ratio cant be 0)
# lines[:, L['tau']] = torch.where(lines[:, L['tau']] == 0, 1, lines[:, L['tau']])

# generators = torch.tensor(gen_data[:, [0, 8, 9, 1, 5, 2]], dtype=torch.float32)
G = {'bus_i': 0, 'Pmax': 1, 'Pmin': 2, 'Pg': 3, 'vg': 4, 'Qg': 5}  # indices of generator data
# generators[generators_data[:, G['bus_i']].int() - 1] = generators_data  # -1 as bus indices start from 1
# generators[:, G['bus_i']] = torch.tensor(range(1, buses.shape[0]+1))  # set bus indices to match buses, gen indices stay the same
# del generators_data  # delete to free memory, use generators for same length as buses

# costs = torch.tensor(cost_data, dtype=torch.float32)  # needed?
# cost format: model, startup cost, shutdown cost, nr coefficients, cost coefficients

# x = torch.tensor(buses, dtype=torch.float32)
# edge_index = torch.tensor(lines[:, :2].T, dtype=torch.int16)  # Transpose to match COO format
# edge_attr = torch.tensor(lines[:, 2:], dtype=torch.float32)
# graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def prepare_grid(case_nr, augmentation_nr):
    case_augmented =  pkl.load(open(f'./data/case{case_nr}/augmented_case{case_nr}_{augmentation_nr}.pkl', 'rb'))
    bus_data = torch.tensor(case_augmented['bus'], dtype=torch.float32)
    buses = torch.tensor(bus_data[:, [0, 1, 2, 3, 4, 5]], dtype=torch.float32)
    buses = torch.cat((buses, torch.zeros((buses.shape[0], 1), dtype=torch.float32)), dim=1)  # add qg column for inserting values

    lines_data = torch.tensor(case_augmented['branch'], dtype=torch.float32)
    lines = torch.tensor(lines_data[:, [0, 1, 2, 3, 4, 8, 9]], dtype=torch.float32)
    lines[:, L['tau']] = torch.where(lines[:, L['tau']] == 0, 1, lines[:, L['tau']])

    gen_data = torch.tensor(case_augmented['gen'], dtype=torch.float32)
    generators = torch.tensor(gen_data[:, [0, 8, 9, 1, 5, 2]], dtype=torch.float32)
    return buses, lines, generators

class LearningBlock(nn.Module):  # later change hidden dim to more dims, currently suggested latent=hidden
    def __init__(self, dim_in, hidden_dim, dim_out):
        super(LearningBlock, self).__init__()
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, dim_out)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        x = self.lrelu(x)
        x = self.linear3(x)
        return x


class GNS(nn.Module):
    def __init__(self, latent_dim=10, hidden_dim=10, K=30, gamma=0.9):
        super(GNS, self).__init__()

        self.net = nn.ModuleDict()

        for k in range(K):
            self.net[str(k)] = LearningBlock(dim_in=2, hidden_dim=1, dim_out=1)
        self.K = K

    def forward(self, buses, lines, generators, G):
        input_ = torch.rand(2, 2)
        losses = 0
        out_list = []
        for k in range(self.K):
            for i in range(2):
                out = self.net[str(k)](input_)
                out_list.append(out)
            losses = losses + torch.stack(out_list).sum()
        return '_', '_', losses


#initialization
latent_dim = 10  # increase later
hidden_dim = 10  # increase later
gamma = 0.9
K = 3  # correction updates, 30 in paper, less for debugging

# torch.set_default_device('cuda')
model = GNS(latent_dim=latent_dim, hidden_dim=hidden_dim, K=K)
torch.autograd.set_detect_anomaly(True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

n_runs = 100  # 10**6 used in paper
best_loss = torch.tensor(float('inf'))
best_model = model
loss_increase_counter = 0
print_every = 1
case_nr = 14  # 14, 30, 118, 300
for run in range(n_runs):
    # sample from different grids
    augmentation_nr = np.random.random_integers(0, 9)  # random augmentation of the 10
    buses, lines, generators = prepare_grid(case_nr, augmentation_nr)

    v, theta, loss = model(buses=buses, lines=lines, generators=generators, G=G)
    
    #delta_p, delta_q = local_power_imbalance(v=v, theta=theta, buses=buses, lines=lines, gens=generators)
    
    #last_loss = torch.sum(delta_p**2 + delta_q**2) / buses.shape[0]  # equasion (23)

    # total_loss = torch.sum(loss)
    # total_loss = model.k_loss
    # total_loss = calculate_total_loss(delta_p, delta_q, gamma, K)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    # loss.detach()




