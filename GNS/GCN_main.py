import torch
from torch_scatter import scatter_add
import torch.nn as nn
import random
import pickle as pkl
from torch_geometric.data import Data
# from torch_geometric.nn import MessagePassing
from utils import get_BLG, load_all_grids
from torch.profiler import profile, record_function, ProfilerActivity
# import cProfile
# import wandb


# paper for this whole project, very careful reading: https://pscc-central.epfl.ch/repo/papers/2020/715.pdf
# documentation of data formats for pypower, very careful reading: https://rwl.github.io/PYPOWER/api/pypower.caseformat-module.html
## case format: https://rwl.github.io/PYPOWER/api/pypower.caseformat-module.html

B, L, G = get_BLG()


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

    y_ij = 1 / torch.sqrt(lines[:, L['r']].pow(2) + lines[:, L['x']].pow(2))
    # delta_ij refers to v difference between i and j, not the angle difference
    delta_ij = theta[src] - theta[dst]
    # theta_shift_ij = torch.atan2(lines[:, L['r']], lines[:, L['x']])
    theta_shift_ij = lines[:, L['theta']]
    msg = torch.abs(v[src] * v[dst] * y_ij[src] / lines[:, L['tau']][src] * (torch.sin(theta[src] - theta[dst] - delta_ij[src] - theta_shift_ij[src]) + torch.sin(theta[dst] - theta[src] - delta_ij[src] + theta_shift_ij[src])) + (v[src] / lines[:, L['tau']][src].pow(2)) * y_ij[src] * torch.sin(delta_ij[src]) + v[dst].pow(2) * y_ij[src] * torch.sin(delta_ij[src]))
    aggregated_neighbor_features = scatter_add(msg, dst, out=torch.zeros((buses.shape[0]), dtype=torch.float32), dim=0)
    p_joule = torch.sum(aggregated_neighbor_features)

    p_global = torch.sum(buses[:, B['Pd']]) + torch.sum(v.pow(2) * buses[:, B['Gs']]) + p_joule

    if p_global < gens[:, G['Pg_set']].sum():
        lambda_ = (p_global - gens[:, G['Pmin']].sum()) / (2 * (gens[:, G['Pg_set']].sum() - gens[:, G['Pmin']].sum()))
    else:
        lambda_ = (p_global - 2 * gens[:, G['Pg_set']].sum() + gens[:, G['Pmax']].sum()) / (
                2 * (gens[:, G['Pmax']].sum() - gens[:, G['Pg_set']].sum()))

    if lambda_ < 0.5:  # equasion (21) in paper
        Pg_new = gens[:, G['Pmin']] + 2 * (gens[:, G['Pg_set']] - gens[:, G['Pmin']]) * lambda_
    else:
        Pg_new = 2 * gens[:, G['Pg_set']] - gens[:, G['Pmax']] + 2 * (
                gens[:, G['Pmax']] - gens[:, G['Pg_set']]) * lambda_

    # if Pg is larger than Pmax in any value of the same index, this should be impossible!
    rnd_o1 = torch.rand(1)
    if rnd_o1 < 0.001:
        # if torch.any(Pg_new > gens[:, G['Pmax']]):
        print(f'lambda: {lambda_}')
    qg_new_start = buses[:, B['Qd']] - buses[:, B['Bs']] * v.pow(2)

    delta_ji = theta[dst] - theta[src]
    # theta_shift_ij = torch.atan2(lines[:, L['r']], lines[:, L['x']])
    theta_shift_ij = lines[:, L['theta']]
    msg_from = -v[src] * v[dst] * y_ij[src] / lines[:, L['tau']][src] * torch.cos(
        theta[src] - theta[dst] - delta_ij[src] - theta_shift_ij[src]) + (v[src] / lines[:, L['tau']][src]).pow(2) * (y_ij[src] * torch.cos(delta_ij[src]) - lines[:, L['b']][src] / 2)
    msg_to = -v[dst] * v[src] * y_ij[dst] / lines[:, L['tau']][dst] * torch.cos(
        theta[dst] - theta[src] - delta_ji[dst] - theta_shift_ij[dst]) + v[dst].pow(2) * (
                         y_ij[dst] * torch.sin(delta_ji[dst]) - lines[:, L['b']][dst] / 2)

    aggr_from = scatter_add(msg_from, dst, out=torch.zeros((buses.shape[0]), dtype=torch.float32), dim=0)
    aggr_to = scatter_add(msg_to, src, out=torch.zeros((buses.shape[0]), dtype=torch.float32), dim=0)
    qg_new = qg_new_start - aggr_from - aggr_to

    return Pg_new, qg_new

def local_power_imbalance(v, theta, buses, lines, gens, pg_k, qg_k, B, L, G):
    delta_p_gens = scatter_add(pg_k, torch.tensor(gens[:, G['bus_i']] - 1, dtype=torch.int64), out=torch.zeros((buses.shape[0]), dtype=torch.float32), dim=0)
    delta_p_start = delta_p_gens - buses[:, B['Pd']] - buses[:, B['Gs']] * v.pow(2)
    delta_q_start = qg_k - buses[:, B['Qd']] + buses[:, B['Bs']] * v.pow(2)

    src = torch.tensor((lines[:, L['f_bus']]).int() - 1, dtype=torch.int64)
    dst = torch.tensor((lines[:, L['t_bus']]).int() - 1, dtype=torch.int64)
    y_ij = 1 / torch.sqrt(lines[:, L['r']].pow(2) + lines[:, L['x']].pow(2))
    delta_ij = theta[src] - theta[dst]
    delta_ji = theta[dst] - theta[src]
    # theta_shift_ij = torch.atan2(lines[:, L['r']], lines[:, L['x']])
    theta_shift_ij = lines[:, L['theta']]
    p_msg_from = v[src] * v[dst] * y_ij[src] / lines[:, L['tau']][src] * torch.sin(theta[src] - theta[dst] - delta_ij[src] - theta_shift_ij[src]) + (v[src] / lines[:, L['tau']][src]).pow(2) * y_ij[src] * torch.sin(delta_ij[src])
    p_msg_to = v[dst] * v[src] * y_ij[dst] / lines[:, L['tau']][dst] * torch.sin(theta[dst] - theta[src] - delta_ji[dst] - theta_shift_ij[dst]) + v[dst].pow(2) * y_ij[dst] * torch.sin(delta_ji[dst])

    p_sum_from = scatter_add(p_msg_from, dst, out=torch.zeros((buses.shape[0]), dtype=torch.float32), dim=0)
    p_sum_to = scatter_add(p_msg_to, src, out=torch.zeros((buses.shape[0]), dtype=torch.float32), dim=0)
    delta_p = delta_p_start + p_sum_from + p_sum_to

    q_msg_from = -v[src] * v[dst] * y_ij[src] / lines[:, L['tau']][src] * torch.cos(theta[src] - theta[dst] - delta_ij[src] - theta_shift_ij[src]) + (v[src] / lines[:, L['tau']][src]).pow(2) * (y_ij[src] * torch.cos(delta_ij[src]) - lines[:, L['b']][src] / 2)
    q_msg_to = -v[dst] * v[src] * y_ij[dst] / lines[:, L['tau']][dst] * torch.cos(theta[dst] - theta[src] - delta_ji[dst] - theta_shift_ij[dst]) + v[dst].pow(2) * (y_ij[dst] * torch.sin(delta_ji[dst]) - lines[:, L['b']][dst] / 2)  # last cos is sin in paper??? Shouldnt be true as the complex power is with cos

    q_sum_from = scatter_add(q_msg_from, dst, out=torch.zeros((buses.shape[0]), dtype=torch.float32), dim=0)
    q_sum_to = scatter_add(q_msg_to, src, out=torch.zeros((buses.shape[0]), dtype=torch.float32), dim=0)
    delta_q = delta_q_start + q_sum_from + q_sum_to
    # print(f'delta_p: {delta_p}')
    return delta_p, delta_q


class GNS(nn.Module):
    def __init__(self, latent_dim=10, hidden_dim=10, K=30, gamma=0.9):
        super(GNS, self).__init__()

        self.phi_v = nn.ModuleDict()
        self.phi_theta = nn.ModuleDict()
        self.phi_m = nn.ModuleDict()

        self.L_theta = nn.ModuleDict()
        self.L_v = nn.ModuleDict()
        self.L_m = nn.ModuleDict()

        for k in range(K):
            self.phi_v[str(k)] = LearningBlock(5 + latent_dim, hidden_dim, 1)
            self.phi_theta[str(k)] = LearningBlock(5 + latent_dim, hidden_dim, 1)
            self.phi_m[str(k)] = LearningBlock(5 + latent_dim, hidden_dim, 1)

            self.L_theta[str(k)] = LearningBlock(dim_in=5 + latent_dim, hidden_dim=hidden_dim, dim_out=1)
            self.L_v[str(k)] = LearningBlock(dim_in=5 + latent_dim, hidden_dim=hidden_dim, dim_out=1)
            self.L_m[str(k)] = LearningBlock(dim_in=5 + latent_dim, hidden_dim=hidden_dim, dim_out=latent_dim)

        self.latent_dim = latent_dim
        self.gamma = gamma
        self.K = K

    def forward(self, buses, lines, generators, B, L, G):
        # edge_index = torch.tensor(lines[:, :2].t().long(), dtype=torch.long)
        # edge_attr = lines[:, 2:].t()
        # x = buses[:, 1:]
        # data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        m = torch.zeros((buses.shape[0], self.latent_dim), dtype=torch.float32)
        v = torch.ones((buses.shape[0]), dtype=torch.float32)
        theta = torch.zeros((buses.shape[0]), dtype=torch.float32)
        total_loss = 0.
        # make dtype=int64 for scatter_add
        bus_i = torch.tensor(generators[:, G['bus_i']] - 1, dtype=torch.int64)
        v[generators[:, G['bus_i']].long() - 1] = generators[:, G['vg']]
        pg_new = scatter_add(generators[:, G['Pg']], bus_i, out=torch.zeros((buses.shape[0]), dtype=torch.float32), dim=0)
        delta_p = pg_new - buses[:, B['Pd']] - buses[:, B['Gs']] * v.pow(2)
        qg_new = scatter_add(generators[:, G['qg']], bus_i, out=torch.zeros((buses.shape[0]), dtype=torch.float32), dim=0)
        delta_q = qg_new - buses[:, B['Qd']] + buses[:, B['Bs']] * v.pow(2)
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
            phi_v_input = self.phi_v[str(k)](phi_from_input)
            phi_theta_input = self.phi_theta[str(k)](phi_from_input)
            phi_m_input = self.phi_m[str(k)](phi_from_input)

            phi_v_sum = scatter_add(phi_v_input, dst, out=torch.zeros((buses.shape[0], 1), dtype=torch.float32), dim=0)
            phi_theta_sum = scatter_add(phi_theta_input, dst, out=torch.zeros((buses.shape[0], 1), dtype=torch.float32), dim=0)
            phi_m_sum = scatter_add(phi_m_input, dst, out=torch.zeros((buses.shape[0], 1), dtype=torch.float32), dim=0)

            network_input_v = torch.cat((v.unsqueeze(1), theta.unsqueeze(1), delta_p.unsqueeze(1), delta_q.unsqueeze(1), m, phi_v_sum), dim=1)
            network_input_theta = torch.cat((v.unsqueeze(1), theta.unsqueeze(1), delta_p.unsqueeze(1), delta_q.unsqueeze(1), m, phi_theta_sum), dim=1)
            network_input_m = torch.cat((v.unsqueeze(1), theta.unsqueeze(1), delta_p.unsqueeze(1), delta_q.unsqueeze(1), m, phi_m_sum), dim=1)

            theta_update = self.L_theta[str(k)](network_input_theta)
            theta = theta + theta_update.squeeze()

            v_update = self.L_v[str(k)](network_input_v)
            non_gens_mask = torch.ones_like(v, dtype=torch.bool)
            non_gens_mask[generators[:, G['bus_i']].long() - 1] = False
            v = torch.where(non_gens_mask, v + v_update.squeeze(), v)

            m_update = self.L_m[str(k)](network_input_m)
            m = m + m_update

            Pg_new, qg_new = global_active_compensation(v, theta, buses, lines, generators, B, L, G)

            delta_p, delta_q = local_power_imbalance(v, theta, buses, lines, generators, Pg_new, qg_new, B, L, G)

            rng_o1 = torch.rand(1)
            if rng_o1 < self.K/3000 and k == self.K:
                print(f'delta_p: {delta_p.data[:7]}')
                print(f'delta_q: {delta_q.data[:7]}')
            total_loss = total_loss + self.gamma**(self.K - k) * torch.sum(delta_p.pow(2) + delta_q.pow(2)).div(buses.shape[0])
        return v, theta, total_loss


def main():
    # Weights and Biases package
    # wandb.login()

    # initialization
    latent_dim = 10  # increase later
    hidden_dim = 10  # increase later
    gamma = 0.9
    K = 2  # correction updates, 30 in paper, less for debugging

    model = GNS(latent_dim=latent_dim, hidden_dim=hidden_dim, K=K, gamma=gamma)

    ## comment CUDA code below for CPU usage
    # if torch.cuda.is_available():
    #     torch.set_default_device('cuda')
    # model.to(torch.device('cuda'))

    epochs = 10**6  # 10**6 used in paper
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = torch.tensor(float('inf'))
    loss_increase_counter = 0
    case_nr = 14  # 14, 30, 118, 300
    batch_size = 2**7  # 2**7==128
    print_every = 1
    nr_samples = 2**10  # 2**10==1024
    all_buses, all_lines, all_generators = load_all_grids(case_nr, nr_samples=nr_samples)

    # wandb_run = wandb.init(
    #     project="GNS",
    #     config={
    #         "learning_rate": lr,
    #         "epochs": epochs,
    #         "batch_size": batch_size,
    #         "latent_dim": latent_dim,
    #         "hidden_dim": hidden_dim,
    #         "case_nr": case_nr,
    #         "K": K,
    #         "nr_samples": nr_samples,
    #     },
    # )

    for epoch in range(epochs):
        epoch_losses = torch.zeros(nr_samples // batch_size, dtype=torch.float64)
        for batch_idx_start in range(0, nr_samples, batch_size):
            losses = torch.zeros(batch_size, dtype=torch.float64)
            for i in range(batch_idx_start, batch_idx_start + batch_size):
                buses, lines, generators = all_buses[i], all_lines[i], all_generators[i]
                v, theta, loss = model(buses=buses, lines=lines, generators=generators, B=B, L=L, G=G)
                losses[i % batch_size] = loss  # -1 because i starts at 1 because 0 is not augmented

            total_loss = torch.mean(losses)
            epoch_losses[batch_idx_start // batch_size] = total_loss.data
            # wandb.log({"epoch": epoch, "loss": total_loss})
            # print(f'Run: {epoch}, Batch Loss: {total_loss}')
            total_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        epoch_loss = torch.mean(epoch_losses)
        if epoch_loss >= best_loss:
            loss_increase_counter += 1
            if loss_increase_counter > 100:
                print('Loss is increasing')
                break
        else:
            best_loss = epoch_loss
            best_model = model
            loss_increase_counter = 0
            torch.save(best_model.state_dict(), f'../models/best_model_c{case_nr}_K{K}_L{latent_dim}_H{hidden_dim}.pth')
        if epoch % print_every == 0:
            print(f'Run: {epoch}, Epoch Loss: {epoch_loss}, best loss: {best_loss}')


if __name__ == '__main__':
    ## pytorch profiler
    # with profile(activities=[ProfilerActivity.CUDA]) as prof:
    #     with record_function("model_inference"):
    #         main()
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    main()

