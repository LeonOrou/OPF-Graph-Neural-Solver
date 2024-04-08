import torch
from torch_scatter import scatter_add
from torch import index_add
import torch.nn as nn
import random
import pickle as pkl
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
# import wandb


# paper for this whole project, very careful reading: https://pscc-central.epfl.ch/repo/papers/2020/715.pdf
# documentation of data formats for pypower, very careful reading: https://rwl.github.io/PYPOWER/api/pypower.caseformat-module.html

# ppopt = ppoption(PF_ALG=1)
# r = runpf(pf_net_pyp, ppopt)
# printpf(r)

# num_nodes = bus_data.shape[0]
# num_edges = branch_data.shape[0]

## case format: https://rwl.github.io/PYPOWER/api/pypower.caseformat-module.html

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
    buses[:, [4, 5]] /= baseMV  # normalize Gs and Bs to gs and bs by dividing by baseMV
    buses = torch.cat((buses, torch.zeros((buses.shape[0], 1), dtype=torch.float32)), dim=1)  # add qg column for inserting values
    # Normalizing the Power P, Q and ?S
    buses[:, [2, 3, 6]] /= baseMV

    lines_data = torch.tensor(case_augmented['branch'], dtype=torch.float32)
    lines = torch.tensor(lines_data[:, [0, 1, 2, 3, 4, 8, 9]], dtype=torch.float32)
    lines[:, L['tau']] = torch.where(lines[:, L['tau']] == 0, 1, lines[:, L['tau']])

    gen_data = torch.tensor(case_augmented['gen'], dtype=torch.float32)
    generators = torch.tensor(gen_data[:, [0, 8, 9, 1, 5, 2]], dtype=torch.float32)
    generators = torch.cat((generators, generators[:, 3].unsqueeze(dim=1)), dim=1)  # add changable Pg and leave original Pg as Pg_set
    # Normalizing the Power P, Q
    generators[:, [1, 2, 3, 5, 6]] /= baseMV
    return buses, lines, generators


class LearningBlock(nn.Module):  # later change hidden dim to more dims, currently suggested latent=hidden
    def __init__(self, dim_in, hidden_dim, dim_out):
        super(LearningBlock, self).__init__()
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, dim_out)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        # x = self.lrelu(x)
        # x = self.linear3(x)
        x = self.lrelu(x)
        x = self.linear4(x)
        return x


def global_active_compensation(v, theta, buses, lines, gens, B, L, G):
    src = torch.tensor((lines[:, L['f_bus']]).int() - 1, dtype=torch.int64)
    dst = torch.tensor((lines[:, L['t_bus']]).int() - 1, dtype=torch.int64)

    y_ij = 1 / torch.sqrt(lines[:, L['r']] ** 2 + lines[:, L['x']] ** 2)
    # delta_ij refers to v difference between i and j, not the angle difference
    delta_ij = theta[src] - theta[dst]
    theta_shift_ij = torch.atan2(lines[:, L['r']], lines[:, L['x']])
    msg = torch.abs(v[src] * v[dst] * y_ij / lines[:, L['tau']] * (torch.sin(theta[src] - theta[dst] - delta_ij - theta_shift_ij) + torch.sin(theta[dst] - theta[src] - delta_ij + theta_shift_ij)) + (v[src] / lines[:, L['tau']] ** 2) * y_ij * torch.sin(delta_ij) + v[dst] ** 2 * y_ij * torch.sin(delta_ij))
    aggregated_neighbor_features = scatter_add(msg, dst, out=torch.zeros((buses.shape[0]), dtype=torch.float32), dim=0)
    p_joule = torch.sum(aggregated_neighbor_features)

    pd_sum = torch.sum(buses[:, B['Pd']])
    gs_sum = torch.sum(buses[:, B['Gs']])
    # TODO: change p_global computation such that it is a torch.scatter_add_() operation
    p_global = torch.sum(v.pow(2)) * gs_sum + pd_sum + p_joule

    if p_global < gens[:, G['Pg_set']].sum():
        lambda_ = (p_global - gens[:, G['Pmin']].sum()) / (2 * (gens[:, G['Pg_set']].sum() - gens[:, G['Pmin']].sum()))
    else:
        lambda_ = (p_global - 2 * gens[:, G['Pg_set']].sum() + gens[:, G['Pmax']].sum()) / (
                2 * (gens[:, G['Pmax']].sum() - gens[:, G['Pg_set']].sum()))
    # if lambda_ > 1:
    #     lambda_ = 1

    if lambda_ < 0.5:  # equasion (21) in paper
        Pg_new = gens[:, G['Pmin']] + 2 * (gens[:, G['Pg_set']] - gens[:, G['Pmin']]) * lambda_
    else:
        Pg_new = 2 * gens[:, G['Pg_set']] - gens[:, G['Pmax']] + 2 * (
                gens[:, G['Pmax']] - gens[:, G['Pg_set']]) * lambda_

    # if Pg is larger than Pmax in any value of the same index, this should be impossible!
    rnd_o1 = torch.rand(1)
    if rnd_o1 < 0.01:
        # if torch.any(Pg_new > gens[:, G['Pmax']]):
        print(f'lambda: {lambda_}')
    qg_new_start = buses[:, B['Qd']] - buses[:, B['Bs']] * v ** 2
    src = torch.tensor((lines[:, L['f_bus']]).int() - 1, dtype=torch.int64)
    dst = torch.tensor((lines[:, L['t_bus']]).int() - 1, dtype=torch.int64)

    y_ij = 1 / torch.sqrt(lines[:, L['r']] ** 2 + lines[:, L['x']] ** 2)
    delta_ij = v[src] - v[dst]
    theta_shift_ij = torch.atan2(lines[:, L['r']], lines[:, L['x']])
    msg_from = -v[src] * v[dst] * y_ij[src] / lines[:, L['tau']] * torch.cos(
        theta[src] - theta[dst] - delta_ij[src] - theta_shift_ij) + (v[src] / lines[:, L['tau']]) ** 2 * (
                           y_ij[src] * torch.cos(delta_ij[src]) - lines[:, L['b']] / 2)
    msg_to = -v[dst] * v[src] * y_ij[dst] / lines[:, L['tau']] * torch.cos(
        theta[dst] - theta[src] - delta_ij[dst] - theta_shift_ij) + v[dst] ** 2 * (
                         y_ij[dst] * torch.sin(delta_ij[dst]) - lines[:, L['b']] / 2)

    aggr_from = scatter_add(msg_from, dst, out=torch.zeros((buses.shape[0]), dtype=torch.float32), dim=0)
    aggr_to = scatter_add(msg_to, src, out=torch.zeros((buses.shape[0]), dtype=torch.float32), dim=0)
    qg_new = qg_new_start - aggr_from - aggr_to

    return Pg_new, qg_new

def local_power_imbalance(v, theta, buses, lines, gens, pg_k, qg_k, B, L, G):
    delta_p_base = -buses[:, B['Pd']] - buses[:, B['Gs']] * v ** 2
    # delta_p_gens = [gens[:, G['Pg']][gens[:, G['bus_i']].int() - 1 == i] if i in gens[:, G['bus_i']].int() - 1 else 0. for i in range(buses.shape[0])]
    delta_p_gens = [pg_k[gens[:, G['bus_i']].int() - 1 == i] if i in gens[:, G['bus_i']].int() - 1 else 0.
                    for i in range(buses.shape[0])]
    delta_p_start = delta_p_base + torch.tensor(delta_p_gens)
    # delta_q_start = buses[:, B['qg']] - buses[:, B['Qd']] - buses[:, B['Bs']] * v**2
    delta_q_start = qg_k - buses[:, B['Qd']] - buses[:, B['Bs']] * v ** 2

    # TODO: change delta_p and delta_q computation such that it is a torch.scatter_add_() operation
    src = torch.tensor((lines[:, L['f_bus']]).int() - 1, dtype=torch.int64)
    dst = torch.tensor((lines[:, L['t_bus']]).int() - 1, dtype=torch.int64)
    y_ij = 1 / torch.sqrt(lines[:, L['r']] ** 2 + lines[:, L['x']] ** 2)
    delta_ij = theta[src] - theta[dst]
    theta_shift_ij = torch.atan2(lines[:, L['r']], lines[:, L['x']])

    p_msg_from = v[src] * v[dst] * y_ij[src] / lines[:, L['tau']] * torch.sin(theta[src] - theta[dst] - delta_ij[src] - theta_shift_ij) + (v[src] / lines[:, L['tau']]) ** 2 * y_ij[src] * torch.sin(delta_ij[src])
    p_msg_to = v[dst] * v[src] * y_ij[dst] / lines[:, L['tau']] * torch.sin(theta[dst] - theta[src] - delta_ij[dst] - theta_shift_ij) + v[dst] ** 2 * y_ij[dst] * torch.sin(delta_ij[dst])

    p_sum_from = scatter_add(p_msg_from, dst, out=torch.zeros((buses.shape[0]), dtype=torch.float32), dim=0)
    p_sum_to = scatter_add(p_msg_to, src, out=torch.zeros((buses.shape[0]), dtype=torch.float32), dim=0)
    delta_p = delta_p_start + p_sum_from + p_sum_to

    q_msg_from = -v[src] * v[dst] * y_ij[src] / lines[:, L['tau']] * torch.cos(theta[src] - theta[dst] - delta_ij[src] - theta_shift_ij) + (v[src] / lines[:, L['tau']]) ** 2 * (y_ij[src] * torch.cos(delta_ij[src]) - lines[:, L['b']] / 2)
    q_msg_to = -v[dst] * v[src] * y_ij[dst] / lines[:, L['tau']] * torch.sin(theta[dst] - theta[src] - delta_ij[dst] - theta_shift_ij) + v[dst] ** 2 * (y_ij[dst] * torch.cos(delta_ij[dst]) - lines[:, L['b']] / 2)  # last cos is sin in paper??? Shouldnt be true as the complex power is with cos

    q_sum_from = scatter_add(q_msg_from, dst, out=torch.zeros((buses.shape[0]), dtype=torch.float32), dim=0)
    q_sum_to = scatter_add(q_msg_to, src, out=torch.zeros((buses.shape[0]), dtype=torch.float32), dim=0)
    delta_q = delta_q_start + q_sum_from + q_sum_to
    # print(f'delta_p: {delta_p}')
    return delta_p, delta_q


class GNS(nn.Module):
    def __init__(self, latent_dim=10, hidden_dim=10, K=30, gamma=0.9):
        super(GNS, self).__init__()
        # self.correction_block = nn.ModuleDict()

        self.phi_from = nn.ModuleDict()
        # self.phi_to = nn.ModuleDict()
        self.phi_loop = nn.ModuleDict()
        self.correction_block = nn.ModuleDict()
        self.D = nn.ModuleDict()
        self.L_theta = nn.ModuleDict()
        self.L_v = nn.ModuleDict()
        self.L_m = nn.ModuleDict()

        for k in range(K):
            self.phi_from[str(k)] = LearningBlock(latent_dim + 5, hidden_dim, latent_dim)
            # self.phi_to[str(k)] = LearningBlock(latent_dim + 5, hidden_dim, latent_dim)
            self.phi_loop[str(k)] = LearningBlock(latent_dim + 5, hidden_dim, latent_dim)

            self.L_theta[str(k)] = LearningBlock(dim_in=4 + 2 * latent_dim, hidden_dim=hidden_dim, dim_out=1)
            self.L_v[str(k)] = LearningBlock(dim_in=4 + 2 * latent_dim, hidden_dim=hidden_dim, dim_out=1)
            self.L_m[str(k)] = LearningBlock(dim_in=4 + 2 * latent_dim, hidden_dim=hidden_dim, dim_out=latent_dim)

        self.latent_dim = latent_dim
        self.gamma = gamma
        self.K = K

    def forward(self, buses, lines, generators, B, L, G):
        # TODO: normalizing all data by subtracting mean and dividing by std
        # buses_mean = torch.mean(buses[:, 1:], dim=0)
        # lines_mean = torch.mean(lines[:, 2:], dim=0)
        # generators_mean = torch.mean(generators[:, 1:], dim=0)
        # buses[:, 1:] = buses[:, 1:] - buses_mean
        # # buses[:, 1:] = buses[:, 1:] / torch.std(buses[:, 1:], dim=0)
        # lines[:, 2:] = lines[:, 2:] - lines_mean
        # # lines[:, 2:] = lines[:, 2:] / torch.std(lines[:, 2:], dim=0)
        # generators[:, 1:] = generators[:, 1:] - generators_mean
        # # generators[:, 1:] = generators[:, 1:] / torch.std(generators[:, 1:], dim=0)
        # buses = torch.nan_to_num(buses)
        # lines = torch.nan_to_num(lines)
        # generators = torch.nan_to_num(generators)
        ### Normalizing Data
        # buses_mean = torch.mean(buses[:, [2, 3]], dim=0)
        # lines_mean = torch.mean(lines[:, [2, 3, 4, 6]], dim=0)
        # generators_mean = torch.mean(generators[:, [1, 3, 4, 5, 6]], dim=0)
        # buses_std = torch.std(buses[:, [2, 3]], dim=0)
        # lines_std = torch.std(lines[:, [2, 3, 4, 6]], dim=0)
        # generators_std = torch.std(generators[:, [1, 3, 4, 5, 6]], dim=0)
        # buses[:, [2, 3]] = (buses[:, [2, 3]] - buses_mean) / buses_std
        # lines[:, [2, 3, 4, 6]] = (lines[:, [2, 3, 4, 6]] - lines_mean) / lines_std
        # generators[:, [1, 3, 4, 5, 6]] = (generators[:, [1, 3, 4, 5, 6]] - generators_mean) / generators_std

        alpha = 1/self.K
        edge_index = torch.tensor(lines[:, :2].t().long(), dtype=torch.long)
        edge_attr = lines[:, 2:].t()
        x = buses[:, 1:]
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        m = torch.zeros((buses.shape[0], self.latent_dim), dtype=torch.float32)
        v = torch.ones((buses.shape[0]), dtype=torch.float32)
        theta = torch.zeros((buses.shape[0]), dtype=torch.float32)
        total_loss = 0.

        v[generators[:, G['bus_i']].long() - 1] = generators[:, G['vg']]
        delta_p = -buses[:, B['Pd']] - buses[:, B['Gs']] * v.pow(2)
        delta_p[generators[:, G['bus_i']].long() - 1] = delta_p[generators[:, G['bus_i']].long() - 1] + generators[:, G['Pg']]
        delta_q = buses[:, B['qg']] - buses[:, B['Qd']] - buses[:, B['Bs']] * v.pow(2)
        # num_nodes = 100
        # num_edges = 1000
        # in_dim = 32
        # out_dim = 64
        # msg_lin = torch.nn.Linear(in_dim, in_dim)
        # outer_lin = torch.nn.Linear(2*in_dim, out_dim)
        # x = torch.randn(num_nodes, in_dim)
        # edge_index = torch.randint(0, num_nodes, (2, num_edges))
        # src = edge_index[0]
        # dst = edge_index[1]
        # msg = msg_lin(x)[src]
        # aggregated_neighbor_features = scatter_add(msg, dst, out=torch.zeros_like(x), dim=0)
        # out = outer_lin(torch.cat((x, aggregated_neighbor_features), dim=1))
        src = lines[:, 0].long() - 1  # Compute i and j for all lines at once
        dst = lines[:, 1].long() - 1
        # loop_mask = torch.eq(src, dst)
        # src = torch.cat((src, dst), dim=0)
        # dst = torch.cat((dst, src[:math.ceil(len(src)/2)]), dim=0)
        for k in range(self.K):
            # TODO: make phi message as sum of messages from all !direct! neighbors
            # TODO: make phi messages like in github, the delta p and q seems to work but the messages are bullshit
            phi_from_input = torch.cat((m[dst], lines[:, 2:]), dim=1)
            # phi_to_input = torch.cat((m[dst], lines[:, 2:]), dim=1) * (1-loop_mask)
            # phi_loop_input = torch.cat((m[dst], lines[:, 2:]), dim=1)
            # ?only "phi_from" because we collect information only "from" neighbors
            phi_from_input = self.phi_from[str(k)](phi_from_input)
            phi_from_sum = scatter_add(phi_from_input, src, out=torch.zeros((buses.shape[0], self.latent_dim), dtype=torch.float32), dim=0)
            # phi_to_sum = scatter_add(phi_res_v, dst, out=torch.zeros((buses.shape[0], self.latent_dim), dtype=torch.float32), dim=0)
            # phi_loop_sum = scatter_add(phi_res_m, dst, out=torch.zeros((buses.shape[0], self.latent_dim), dtype=torch.float32), dim=0)

            network_input = torch.cat((v.unsqueeze(1), theta.unsqueeze(1), delta_p.unsqueeze(1), delta_q.unsqueeze(1), m, phi_from_sum), dim=1)

            theta_update = self.L_theta[str(k)](network_input)
            theta = theta + theta_update.squeeze() * alpha

            v_update = self.L_v[str(k)](network_input)
            non_gens_mask = torch.ones_like(v, dtype=torch.bool)
            non_gens_mask[generators[:, G['bus_i']].long() - 1] = False
            v = torch.where(non_gens_mask, v + v_update.squeeze() * alpha, v)

            m_update = self.L_m[str(k)](network_input)
            m = m + m_update * alpha

            Pg_new, qg_new = global_active_compensation(v, theta, buses, lines, generators, B, L, G)
            # no matrix afterwards as Pg is last column
            # generators = torch.cat((generators[:, :G['Pg']], Pg_new.unsqueeze(dim=1)), dim=1)
            # buses = torch.cat((buses[:, :B['qg']], qg_new.unsqueeze(dim=1), buses[:, B['qg'] + 1:]), dim=1)
            delta_p, delta_q = local_power_imbalance(v, theta, buses, lines, generators, Pg_new, qg_new, B, L, G)

            total_loss = total_loss + self.gamma**(self.K - k) * torch.sum(delta_p.pow(2) + delta_q.pow(2)) / buses.shape[0]
        return v, theta, total_loss

# Weights and Biases package
# wandb.login()


# initialization
latent_dim = 10  # increase later
hidden_dim = 20  # increase later
gamma = 0.9
K = 10  # correction updates, 30 in paper, less for debugging
if torch.cuda.is_available():
    torch.set_default_device('cuda')
model = GNS(latent_dim=latent_dim, hidden_dim=hidden_dim, K=K)
torch.autograd.set_detect_anomaly(True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_runs = 10 ** 6  # 10**6 used in paper
best_loss = torch.tensor(float('inf'))
best_model = model
loss_increase_counter = 0
print_every = 1
case_nr = 14  # 14, 30, 118, 300
for run in range(n_runs):
    # sample from different grids
    augmentation_nr = random.randint(1, 10)  # random augmentation of the 10
    # augmentation_nr = 1  # 0 is not modified case
    buses, lines, generators = prepare_grid(case_nr, augmentation_nr)

    v, theta, loss = model(buses=buses, lines=lines, generators=generators, B=B, L=L, G=G)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if loss >= best_loss:
        loss_increase_counter += 1
        if loss_increase_counter > 100:
            print('Loss is increasing')
            break
    else:
        best_loss = loss.data
        best_model = model
        loss_increase_counter = 0
        torch.save(best_model.state_dict(), f'../models/best_model_c{case_nr}_K{K}_L{latent_dim}_H{hidden_dim}.pth')
    if run % print_every == 0:
        print(f'Run: {run}, Loss: {loss}, best loss: {best_loss}')



