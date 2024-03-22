import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import random
import pickle as pkl
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_BLG():
    # add column qg to buses with default value 0 as we later use qg for each bus
    B = {'bus_i': 0, 'type': 1, 'Pd': 2, 'Qd': 3, 'Gs': 4, 'Bs': 5, 'qg': 6}  # indices of bus data

    # lines = torch.tensor(branch_data[:, [0, 1, 2, 3, 4, 8, 9]], dtype=torch.float32)
    L = {'f_bus': 0, 't_bus': 1, 'r': 2, 'x': 3, 'b': 4, 'tau': 5, 'theta': 6}  # indices of branch data
    # if tau=0, set it to 1: 0 is default in pypower, but matpower default is 1 (ratio cant be 0)
    # lines[:, L['tau']] = torch.where(lines[:, L['tau']] == 0, 1, lines[:, L['tau']])

    # generators = torch.tensor(gen_data[:, [0, 8, 9, 1, 5, 2]], dtype=torch.float32)
    G = {'bus_i': 0, 'Pmax': 1, 'Pmin': 2, 'Pg_set': 3, 'vg': 4, 'qg': 5, 'Pg': 6}  # indices of generator data
    # generators[generators_data[:, G['bus_i']].int() - 1] = generators_data  # -1 as bus indices start from 1
    # generators[:, G['bus_i']] = torch.tensor(range(1, buses.shape[0]+1))  # set bus indices to match buses, gen indices stay the same
    # del generators_data  # delete to free memory, use generators for same length as buses

    # costs = torch.tensor(cost_data, dtype=torch.float32)  # needed?
    # cost format: model, startup cost, shutdown cost, nr coefficients, cost coefficients
    return B, L, G


B, L, G = get_BLG()


def prepare_grid(case_nr, augmentation_nr):
    case_augmented = pkl.load(open(f'../data/case{case_nr}/augmented_case{case_nr}_{augmentation_nr}.pkl', 'rb'))
    bus_data = torch.tensor(case_augmented['bus'], dtype=torch.float32)
    buses = torch.tensor(bus_data[:, [0, 1, 2, 3, 4, 5]], dtype=torch.float32)
    # Gs and Bs have defaults of 1 in paper, but 0 in matpower
    # Bs is not everywhere 0, but in paper it is everywhere 1 p.u. (of the Qd?)
    buses[:, 4] = buses[:, 3]
    buses[:, 5] = buses[:, 3]
    baseMV = 100  # set to 100 in GitHub, no default in Matpower
    buses[:, 4] /= baseMV  # normalize Gs and Bs to gs and bs by dividing by baseMV
    buses[:, 5] /= baseMV
    buses = torch.cat((buses, torch.zeros((buses.shape[0], 1), dtype=torch.float32)),
                      dim=1)  # add qg column for inserting values

    lines_data = torch.tensor(case_augmented['branch'], dtype=torch.float32)
    lines = torch.tensor(lines_data[:, [0, 1, 2, 3, 4, 8, 9]], dtype=torch.float32)
    lines[:, L['tau']] = torch.where(lines[:, L['tau']] == 0, 1, lines[:, L['tau']])

    gen_data = torch.tensor(case_augmented['gen'], dtype=torch.float32)
    generators = torch.tensor(gen_data[:, [0, 8, 9, 1, 5, 2]], dtype=torch.float32)
    generators = torch.cat((generators, generators[:, 3].unsqueeze(dim=1)),
                           dim=1)  # add changable Pg and leave original Pg as Pg_set
    return buses, lines, generators


class My_GNN_GNN_NN(torch.nn.Module):
    def __init__(self, node_size=None, feat_in=None, feat_size1=None, feat_size2=None, hidden_size1=None,
                 output_size=None):
        super(My_GNN_GNN_NN, self).__init__()
        self.feat_in = feat_in if feat_in is not None else 2
        self.feat_size1 = feat_in if feat_in is not None else 5
        self.feat_size2 = feat_in if feat_in is not None else 4
        self.hidden_size1 = hidden_size1 if hidden_size1 is not None else 38
        self.output_size = output_size if output_size is not None else 18

        self.conv = torch.nn.ModuleDict()
        for k in range(10):
            self.conv[str(k)] = GCNConv(feat_in, feat_in)
        self.conv1 = GCNConv(feat_in, feat_in)
        self.conv2 = GCNConv(feat_in, feat_in)
        self.lin1 = Linear(node_size * feat_size2, hidden_size1)
        self.lin2 = Linear(hidden_size1, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for k in range(10):
            x = self.conv[str(k)](x, edge_index)

        # x = self.conv1(x, edge_index)
        # # x = torch.tanh(x)
        #
        # x = self.conv2(x, edge_index)
        # x = torch.tanh(x)

        # x = x.flatten(start_dim=0)
        # x = self.lin1(x)
        # # x = torch.tanh(x)
        #
        # x = self.lin2(x)

        return x.T

n_bus = 14
feat_in = 2
feat_size1 = 8
feat_size2 = 4
hidden_size1 = 30
output_size = n_bus*2
lr = 0.0001

model = My_GNN_GNN_NN(n_bus, feat_in, feat_size1, feat_size2, hidden_size1, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)



train_loss = 0.
for epoch in range(2001):
    augmentation_nr = random.randint(1, 10)  # random augmentation of the 10
    buses, lines, gens = prepare_grid(case_nr=n_bus, augmentation_nr=augmentation_nr)
    pg_qg = gens[:, [G['Pg_set'], G['qg']]]
    # make pg_qg_bus_size a tensor with size of buses shape[0], with 0s at the indices that are not bus_i in gens
    pg_qg_bus_size = torch.zeros((buses.shape[0], 2), dtype=torch.float32)
    pg_qg_bus_size[gens[:, 0].int()-1] = pg_qg  # -1 as bus indices start from 1
    model.train()
    data = Data(x=pg_qg_bus_size, edge_index=torch.tensor(lines[:, [0, 1]].T-1, dtype=torch.int64))
    power = model(data)
    Pi = power[0]
    Qi = power[1]
    Pd = buses[:, B['Pd']]
    Qd = buses[:, B['Qd']]
    optimizer.zero_grad()
    # Pi = v*torch.sum(v*(B['Gs']*torch.cos(theta) + B['Bs']*torch.sin(theta)))
    # Qi = v*torch.sum(v*(B['Gs']*torch.sin(theta) - B['Bs']*torch.cos(theta)))
    # loss is MSE of every bus' Pg and qg and their real demand
    loss = torch.sum(torch.mean((Pi - Pd)**2) + torch.mean((Qi - Qd)**2))
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
    print(f'Epoch {epoch}: Loss: {train_loss}')

