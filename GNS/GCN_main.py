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
    B = {'bus_i': 0, 'type': 1, 'Pd': 2, 'Qd': 3, 'Gs': 4, 'Bs': 5, 'qg': 6}  # indices of bus data

    L = {'f_bus': 0, 't_bus': 1, 'r': 2, 'x': 3, 'b': 4, 'tau': 5, 'theta': 6}  # indices of branch data

    G = {'bus_i': 0, 'Pmax': 1, 'Pmin': 2, 'Pg_set': 3, 'vg': 4, 'qg': 5, 'Pg': 6}  # indices of generator data

    # costs = torch.tensor(cost_data, dtype=torch.float32)  # needed?
    # cost format: model, startup cost, shutdown cost, nr coefficients, cost coefficients
    return B, L, G


B, L, G = get_BLG()


def prepare_grid(case_nr, augmentation_nr):
    case_augmented = pkl.load(open(f'../data/case{case_nr}/augmented_case{case_nr}_{augmentation_nr}.pkl', 'rb'))
    bus_data = torch.tensor(case_augmented['bus'], dtype=torch.float32)
    lines_data = torch.tensor(case_augmented['branch'], dtype=torch.float32)
    gen_data = torch.tensor(case_augmented['gen'], dtype=torch.float32)
    buses = torch.tensor(bus_data[:, [0, 1, 2, 3, 4, 5]], dtype=torch.float32)
    # Gs and Bs have defaults of 1 in paper, but 0 in matpower
    # Bs is not everywhere 0, but in paper it is everywhere 1 p.u. (of the Qd?)
    buses[:, [4, 5]] = 1.  # Gs and Bs
    baseMV = case_augmented['baseMVA']  # mostly 100
    buses = torch.cat((buses, torch.zeros((buses.shape[0], 1), dtype=torch.float32)), dim=1)  # add qg column for inserting values
    # normalize all P, Q, Gs and Bs to get gs and bs by dividing by baseMV
    buses[:, [2, 3, 4, 5, 6]] /= baseMV

    lines = torch.tensor(lines_data[:, [0, 1, 2, 3, 4, 8, 9]], dtype=torch.float32)
    lines[:, L['tau']] = torch.where(lines[:, L['tau']] == 0, 1, lines[:, L['tau']])

    generators = torch.tensor(gen_data[:, [0, 8, 9, 1, 5, 2]], dtype=torch.float32)
    generators = torch.cat((generators, generators[:, 3].unsqueeze(dim=1)), dim=1)  # copy Pg and concat changable Pg and leave original Pg as Pg_set
    # Normalizing the Power P, Q
    generators[:, [1, 2, 3, 5, 6]] /= baseMV
    return buses, lines, generators

def load_all_grids(case_nr, samples=100):
    all_buses = []
    all_lines = []
    all_generators = []
    for i in range(1, samples+1):  # i==0 is not augmented case, exclude
        buses, lines, generators = prepare_grid(case_nr, i)
        all_buses.append(buses)
        all_lines.append(lines)
        all_generators.append(generators)
    return all_buses, all_lines, all_generators

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
    # theta_shift_ij = torch.atan2(lines[:, L['r']], lines[:, L['x']])
    theta_shift_ij = lines[:, L['theta']]
    msg = torch.abs(v[src] * v[dst] * y_ij[src] / lines[:, L['tau']][src] * (torch.sin(theta[src] - theta[dst] - delta_ij[src] - theta_shift_ij[src]) + torch.sin(theta[dst] - theta[src] - delta_ij[src] + theta_shift_ij[src])) + (v[src] / lines[:, L['tau']][src] ** 2) * y_ij[src] * torch.sin(delta_ij[src]) + v[dst] ** 2 * y_ij[src] * torch.sin(delta_ij[src]))
    aggregated_neighbor_features = scatter_add(msg, dst, out=torch.zeros((buses.shape[0]), dtype=torch.float32), dim=0)
    p_joule = torch.sum(aggregated_neighbor_features)

    p_global = torch.sum(buses[:, B['Pd']]) + torch.sum(v.pow(2) * buses[:, B['Gs']]) + p_joule

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
    if rnd_o1 < 0.005:
        # if torch.any(Pg_new > gens[:, G['Pmax']]):
        print(f'lambda: {lambda_}')
    qg_new_start = buses[:, B['Qd']] - buses[:, B['Bs']] * v ** 2
    src = torch.tensor((lines[:, L['f_bus']]).int() - 1, dtype=torch.int64)
    dst = torch.tensor((lines[:, L['t_bus']]).int() - 1, dtype=torch.int64)

    delta_ji = theta[dst] - theta[src]
    # theta_shift_ij = torch.atan2(lines[:, L['r']], lines[:, L['x']])
    theta_shift_ij = lines[:, L['theta']]
    msg_from = -v[src] * v[dst] * y_ij[src] / lines[:, L['tau']][src] * torch.cos(
        theta[src] - theta[dst] - delta_ij[src] - theta_shift_ij[src]) + (v[src] / lines[:, L['tau']][src]) ** 2 * (y_ij[src] * torch.cos(delta_ij[src]) - lines[:, L['b']][src] / 2)
    msg_to = -v[dst] * v[src] * y_ij[dst] / lines[:, L['tau']][dst] * torch.cos(
        theta[dst] - theta[src] - delta_ji[dst] - theta_shift_ij[dst]) + v[dst] ** 2 * (
                         y_ij[dst] * torch.sin(delta_ji[dst]) - lines[:, L['b']][dst] / 2)

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

    src = torch.tensor((lines[:, L['f_bus']]).int() - 1, dtype=torch.int64)
    dst = torch.tensor((lines[:, L['t_bus']]).int() - 1, dtype=torch.int64)
    y_ij = 1 / torch.sqrt(lines[:, L['r']] ** 2 + lines[:, L['x']] ** 2)
    delta_ij = theta[src] - theta[dst]
    delta_ji = theta[dst] - theta[src]
    # theta_shift_ij = torch.atan2(lines[:, L['r']], lines[:, L['x']])
    theta_shift_ij = lines[:, L['theta']]
    p_msg_from = v[src] * v[dst] * y_ij[src] / lines[:, L['tau']][src] * torch.sin(theta[src] - theta[dst] - delta_ij[src] - theta_shift_ij[src]) + (v[src] / lines[:, L['tau']][src]) ** 2 * y_ij[src] * torch.sin(delta_ij[src])
    p_msg_to = v[dst] * v[src] * y_ij[dst] / lines[:, L['tau']][dst] * torch.sin(theta[dst] - theta[src] - delta_ji[dst] - theta_shift_ij[dst]) + v[dst] ** 2 * y_ij[dst] * torch.sin(delta_ji[dst])

    p_sum_from = scatter_add(p_msg_from, dst, out=torch.zeros((buses.shape[0]), dtype=torch.float32), dim=0)
    p_sum_to = scatter_add(p_msg_to, src, out=torch.zeros((buses.shape[0]), dtype=torch.float32), dim=0)
    delta_p = delta_p_start + p_sum_from + p_sum_to

    q_msg_from = -v[src] * v[dst] * y_ij[src] / lines[:, L['tau']][src] * torch.cos(theta[src] - theta[dst] - delta_ij[src] - theta_shift_ij[src]) + (v[src] / lines[:, L['tau']][src]) ** 2 * (y_ij[src] * torch.cos(delta_ij[src]) - lines[:, L['b']][src] / 2)
    q_msg_to = -v[dst] * v[src] * y_ij[dst] / lines[:, L['tau']][dst] * torch.cos(theta[dst] - theta[src] - delta_ji[dst] - theta_shift_ij[dst]) + v[dst] ** 2 * (y_ij[dst] * torch.sin(delta_ji[dst]) - lines[:, L['b']][dst] / 2)  # last cos is sin in paper??? Shouldnt be true as the complex power is with cos

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
        # self.phi_loop = nn.ModuleDict()
        self.L_theta = nn.ModuleDict()
        self.L_v = nn.ModuleDict()
        self.L_m = nn.ModuleDict()

        for k in range(K):
            self.phi_from[str(k)] = LearningBlock(latent_dim + 5, hidden_dim, latent_dim)
            # self.phi_to[str(k)] = LearningBlock(latent_dim + 5, hidden_dim, latent_dim)
            # self.phi_loop[str(k)] = LearningBlock(latent_dim + 5, hidden_dim, latent_dim)

            self.L_theta[str(k)] = LearningBlock(dim_in=4 + 2 * latent_dim, hidden_dim=hidden_dim, dim_out=1)
            self.L_v[str(k)] = LearningBlock(dim_in=4 + 2 * latent_dim, hidden_dim=hidden_dim, dim_out=1)
            self.L_m[str(k)] = LearningBlock(dim_in=4 + 2 * latent_dim, hidden_dim=hidden_dim, dim_out=latent_dim)

        self.latent_dim = latent_dim
        self.gamma = gamma
        self.K = K

    def forward(self, buses, lines, generators, B, L, G):
        alpha = 1  # update rate from networks for next parameters
        # edge_index = torch.tensor(lines[:, :2].t().long(), dtype=torch.long)
        # edge_attr = lines[:, 2:].t()
        # x = buses[:, 1:]
        # data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

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
        for k in range(self.K):
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
            rng_o1 = torch.rand(1)
            if rng_o1 < self.K/100 and k==K-1:
                print(f'delta_p: {delta_p.data[:7]}')
                print(f'delta_q: {delta_q.data[:7]}')
            total_loss = total_loss + self.gamma**(self.K - k) * torch.sum(delta_p.pow(2) + delta_q.pow(2)) / buses.shape[0]
        return v, theta, total_loss

# Weights and Biases package
# wandb.login()


# initialization
latent_dim = 10  # increase later
hidden_dim = 10  # increase later
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
print_every = 3
case_nr = 14  # 14, 30, 118, 300
batch_size = 5

sample_size = 200
all_buses, all_lines, all_generators = load_all_grids(case_nr, samples=sample_size)

for run in range(n_runs):
    model.train()
    # loop through all batches in one run
    # augmentation_nr = random.randint(1, 1000)  # random augmentation of the 10
    batch_loader = [[i for i in range(j * batch_size, j * batch_size + batch_size)] for j in range(torch.ceil(torch.tensor(sample_size/batch_size)).int())]
    # augmentation_nr = 1  # 0 is not modified case
    run_losses = []
    for batch in batch_loader:
        losses = []
        for grid_i in batch:
            buses, lines, generators = all_buses[grid_i], all_lines[grid_i], all_generators[grid_i]
            v, theta, loss = model(buses=buses, lines=lines, generators=generators, B=B, L=L, G=G)
            losses.append(loss)

        total_loss = sum(losses) / batch_size

        total_loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        run_losses.append(total_loss.data)

    run_loss = sum(run_losses) / len(run_losses)
    if run_loss >= best_loss:
        loss_increase_counter += 1
        if loss_increase_counter > 100:
            print('Loss is increasing')
            break
    else:
        best_loss = run_loss
        best_model = model
        loss_increase_counter = 0
        torch.save(best_model.state_dict(), f'../models/best_model_c{case_nr}_K{K}_L{latent_dim}_H{hidden_dim}_B{batch_size}.pth')
    if run % print_every == 0:
        print(f'Run: {run}, Loss: {run_loss}, best loss: {best_loss}')



