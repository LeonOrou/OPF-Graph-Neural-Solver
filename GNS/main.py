import torch
from torch_scatter import scatter_add
import torch.nn as nn
from utils import get_BLG, load_all_grids
import wandb
import itertools
import time


# paper for this whole project, very careful reading: https://pscc-central.epfl.ch/repo/papers/2020/715.pdf
# documentation of data formats for pypower, very careful reading: https://rwl.github.io/PYPOWER/api/pypower.caseformat-module.html
## case format: https://rwl.github.io/PYPOWER/api/pypower.caseformat-module.html

B, L, G = get_BLG()


class LearningBlock(nn.Module):  # later change hidden dim to more dims, currently suggested latent=hidden
    def __init__(self, dim_in, hidden_dim, dim_out):
        super(LearningBlock, self).__init__()
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, dim_out)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        x = self.lrelu(x)
        x = self.linear4(x)
        return x


def global_active_compensation(v, theta, buses, lines, gens, B, L, G):
    src = torch.tensor((lines[:, L['f_bus']]).int() - 1, dtype=torch.int64)
    dst = torch.tensor((lines[:, L['t_bus']]).int() - 1, dtype=torch.int64)

    y_ij = 1 / torch.sqrt(lines[:, L['r']].pow(2) + lines[:, L['x']].pow(2))
    delta_ij = theta[src] - theta[dst]
    theta_shift_ij = lines[:, L['theta']]
    msg = torch.abs(v[src] * v[dst] * y_ij[src] / lines[:, L['tau']][src] * (torch.sin(theta[src] - theta[dst] - delta_ij[src] - theta_shift_ij[src]) + torch.sin(theta[dst] - theta[src] - delta_ij[src] + theta_shift_ij[src])) + (v[src] / lines[:, L['tau']][src].pow(2)) * y_ij[src] * torch.sin(delta_ij[src]) + v[dst].pow(2) * y_ij[src] * torch.sin(delta_ij[src]))
    aggregated_neighbor_features = scatter_add(msg, dst, out=torch.zeros((buses.shape[0])), dim=0)
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

    ## if Pg is larger than Pmax in any value of the same index, this should be impossible!
    # rnd_o1 = torch.rand(1)
    # if rnd_o1 < 0.0008:
    #     # if torch.any(Pg_new > gens[:, G['Pmax']]):
    #     print(f'lambda: {lambda_}')
    qg_new_start = buses[:, B['Qd']] - buses[:, B['Bs']] * v.pow(2)

    delta_ji = theta[dst] - theta[src]
    theta_shift_ij = lines[:, L['theta']]
    msg_from = -v[src] * v[dst] * y_ij[src] / lines[:, L['tau']][src] * torch.cos(
        theta[src] - theta[dst] - delta_ij[src] - theta_shift_ij[src]) + (v[src] / lines[:, L['tau']][src]).pow(2) * (y_ij[src] * torch.cos(delta_ij[src]) - lines[:, L['b']][src] / 2)
    msg_to = -v[dst] * v[src] * y_ij[dst] / lines[:, L['tau']][dst] * torch.cos(
        theta[dst] - theta[src] - delta_ji[dst] - theta_shift_ij[dst]) + v[dst].pow(2) * (
                         y_ij[dst] * torch.sin(delta_ji[dst]) - lines[:, L['b']][dst] / 2)

    aggr_from = scatter_add(msg_from, dst, out=torch.zeros((buses.shape[0])), dim=0)
    aggr_to = scatter_add(msg_to, src, out=torch.zeros((buses.shape[0])), dim=0)
    qg_new = qg_new_start - aggr_from - aggr_to

    return Pg_new, qg_new

def local_power_imbalance(v, theta, buses, lines, gens, pg_k, qg_k, B, L, G):
    delta_p_gens = scatter_add(pg_k, torch.tensor(gens[:, G['bus_i']] - 1, dtype=torch.int64), out=torch.zeros((buses.shape[0])), dim=0)
    delta_p_start = delta_p_gens - buses[:, B['Pd']] - buses[:, B['Gs']] * v.pow(2)
    delta_q_start = qg_k - buses[:, B['Qd']] + buses[:, B['Bs']] * v.pow(2)

    src = torch.tensor((lines[:, L['f_bus']]).int() - 1, dtype=torch.int64)
    dst = torch.tensor((lines[:, L['t_bus']]).int() - 1, dtype=torch.int64)
    y_ij = 1 / torch.sqrt(lines[:, L['r']].pow(2) + lines[:, L['x']].pow(2))
    delta_ij = theta[src] - theta[dst]
    delta_ji = theta[dst] - theta[src]
    theta_shift_ij = lines[:, L['theta']]
    p_msg_from = v[src] * v[dst] * y_ij[src] / lines[:, L['tau']][src] * torch.sin(theta[src] - theta[dst] - delta_ij[src] - theta_shift_ij[src]) + (v[src] / lines[:, L['tau']][src]).pow(2) * y_ij[src] * torch.sin(delta_ij[src])
    p_msg_to = v[dst] * v[src] * y_ij[dst] / lines[:, L['tau']][dst] * torch.sin(theta[dst] - theta[src] - delta_ji[dst] - theta_shift_ij[dst]) + v[dst].pow(2) * y_ij[dst] * torch.sin(delta_ji[dst])

    p_sum_from = scatter_add(p_msg_from, dst, out=torch.zeros((buses.shape[0])), dim=0)
    p_sum_to = scatter_add(p_msg_to, src, out=torch.zeros((buses.shape[0])), dim=0)
    delta_p = delta_p_start + p_sum_from + p_sum_to

    q_msg_from = -v[src] * v[dst] * y_ij[src] / lines[:, L['tau']][src] * torch.cos(theta[src] - theta[dst] - delta_ij[src] - theta_shift_ij[src]) + (v[src] / lines[:, L['tau']][src]).pow(2) * (y_ij[src] * torch.cos(delta_ij[src]) - lines[:, L['b']][src] / 2)
    q_msg_to = -v[dst] * v[src] * y_ij[dst] / lines[:, L['tau']][dst] * torch.cos(theta[dst] - theta[src] - delta_ji[dst] - theta_shift_ij[dst]) + v[dst].pow(2) * (y_ij[dst] * torch.sin(delta_ji[dst]) - lines[:, L['b']][dst] / 2)  # last cos is sin in paper??? Shouldnt be true as the complex power is with cos

    q_sum_from = scatter_add(q_msg_from, dst, out=torch.zeros((buses.shape[0])), dim=0)
    q_sum_to = scatter_add(q_msg_to, src, out=torch.zeros((buses.shape[0])), dim=0)
    delta_q = delta_q_start + q_sum_from + q_sum_to
    return delta_p, delta_q


class GNS(nn.Module):
    def __init__(self, latent_dim=10, hidden_dim=10, K=30, gamma=0.9, multiple_phi=False):
        super(GNS, self).__init__()

        self.multiple_phis = multiple_phi  # if three different phi networks should be used for each L input or the same

        if self.multiple_phis:
            self.phi_v = nn.ModuleDict()
            self.phi_theta = nn.ModuleDict()
            self.phi_m = nn.ModuleDict()
        else:
            self.phi = nn.ModuleDict()

        self.L_theta = nn.ModuleDict()
        self.L_v = nn.ModuleDict()
        self.L_m = nn.ModuleDict()

        for k in range(K):
            if self.multiple_phis:
                self.phi_v[str(k)] = LearningBlock(5 + latent_dim, hidden_dim, latent_dim)
                self.phi_theta[str(k)] = LearningBlock(5 + latent_dim, hidden_dim, latent_dim)
                self.phi_m[str(k)] = LearningBlock(5 + latent_dim, hidden_dim, latent_dim)
            else:
                self.phi[str(k)] = LearningBlock(5 + latent_dim, hidden_dim, 1)

            self.L_theta[str(k)] = LearningBlock(dim_in=4 + 2 * latent_dim, hidden_dim=hidden_dim, dim_out=1)
            self.L_v[str(k)] = LearningBlock(dim_in=4 + 2 * latent_dim, hidden_dim=hidden_dim, dim_out=1)
            self.L_m[str(k)] = LearningBlock(dim_in=4 + 2 * latent_dim, hidden_dim=hidden_dim, dim_out=latent_dim)

        self.latent_dim = latent_dim
        self.gamma = gamma
        self.K = K

    def forward(self, buses, lines, generators, B, L, G):
        m = torch.zeros((buses.shape[0], self.latent_dim), dtype=torch.float32)
        theta = torch.zeros((buses.shape[0]), dtype=torch.float32)
        total_loss = 0.
        gen_idcs = torch.tensor(generators[:, G['bus_i']] - 1, dtype=torch.int64)
        # make v the gen values, if no gen: 1 v
        v = scatter_add(generators[:, G['vg']], gen_idcs, out=torch.zeros(buses.shape[0]), dim=0)
        v = torch.where(v == 0, torch.ones_like(v), v)
        # v[generators[:, G['bus_i']].long() - 1] = generators[:, G['vg']]
        pg_new = scatter_add(generators[:, G['Pg']], gen_idcs, out=torch.zeros((buses.shape[0]), dtype=torch.float32), dim=0)
        delta_p = pg_new - buses[:, B['Pd']] - buses[:, B['Gs']] * v.pow(2)
        qg_new = scatter_add(generators[:, G['qg']], gen_idcs, out=torch.zeros((buses.shape[0]), dtype=torch.float32), dim=0)
        delta_q = qg_new - buses[:, B['Qd']] + buses[:, B['Bs']] * v.pow(2)
        dst = lines[:, 1].long() - 1
        for k in range(self.K):
            phi_from_input = torch.cat((m[dst], lines[:, 2:]), dim=1)
            if self.multiple_phis:
                phi_v_input = self.phi_v[str(k)](phi_from_input)
                phi_theta_input = self.phi_theta[str(k)](phi_from_input)
                phi_m_input = self.phi_m[str(k)](phi_from_input)

                phi_v_sum = scatter_add(phi_v_input, dst, out=torch.zeros((buses.shape[0], self.latent_dim)), dim=0)
                phi_theta_sum = scatter_add(phi_theta_input, dst, out=torch.zeros((buses.shape[0], self.latent_dim)), dim=0)
                phi_m_sum = scatter_add(phi_m_input, dst, out=torch.zeros((buses.shape[0], self.latent_dim)), dim=0)

                network_input_v = torch.cat((v.unsqueeze(1), theta.unsqueeze(1), delta_p.unsqueeze(1), delta_q.unsqueeze(1), m, phi_v_sum), dim=1)
                network_input_theta = torch.cat((v.unsqueeze(1), theta.unsqueeze(1), delta_p.unsqueeze(1), delta_q.unsqueeze(1), m, phi_theta_sum), dim=1)
                network_input_m = torch.cat((v.unsqueeze(1), theta.unsqueeze(1), delta_p.unsqueeze(1), delta_q.unsqueeze(1), m, phi_m_sum), dim=1)
            else:
                phi_input = self.phi[str(k)](torch.cat((m[dst], lines[:, 2:]), dim=1))
                phi_sum = scatter_add(phi_input, dst, out=torch.zeros((buses.shape[0], self.latent_dim)), dim=0)
                network_input = torch.cat((v.unsqueeze(1), theta.unsqueeze(1), delta_p.unsqueeze(1), delta_q.unsqueeze(1), m, phi_sum), dim=1)

            if self.multiple_phis:
                theta_update = self.L_theta[str(k)](network_input_theta)
                v_update = self.L_v[str(k)](network_input_v)
                m_update = self.L_m[str(k)](network_input_m)
            else:
                theta_update = self.L_theta[str(k)](network_input)
                v_update = self.L_v[str(k)](network_input)
                m_update = self.L_m[str(k)](network_input)

            theta = theta + theta_update.squeeze()

            non_gens_mask = torch.ones_like(v, dtype=torch.bool)
            non_gens_mask[generators[:, G['bus_i']].long() - 1] = False
            v = torch.where(non_gens_mask, v + v_update.squeeze(), v)

            m = m + m_update

            Pg_new, qg_new = global_active_compensation(v, theta, buses, lines, generators, B, L, G)

            delta_p, delta_q = local_power_imbalance(v, theta, buses, lines, generators, Pg_new, qg_new, B, L, G)

            # rng_o1 = torch.rand(1)
            # if rng_o1 < self.K/300 and k == self.K-1:
            #     print(f'delta_p: {delta_p.data[:7]}')
            #     print(f'delta_q: {delta_q.data[:7]}')
            total_loss = total_loss + self.gamma**(self.K - k) * torch.sum(delta_p.pow(2) + delta_q.pow(2)).div(buses.shape[0])
        last_loss = torch.sum(delta_p.pow(2) + delta_q.pow(2)).div(buses.shape[0])
        # v can only be positive and theta between -pi and pi
        v = torch.where(v < 0, torch.zeros_like(v), v)
        return v, theta, total_loss, last_loss


def main():
    # "Weights and Biases" aka wandb model tracking
    wandb.login(key="d234bc98a4761bff39de0e5170df00094ac42269")

    # initialization
    latent_dim = 10  # increase later
    hidden_dim = 10  # increase later
    gamma = 0.9
    K = 15  # correction updates, 30 in paper, less for debugging
    multiple_phi = True
    # params = {'latent_dim': [10, 20],
    #           'hidden_dim': [10],  # TODO: maybe also try more or less hidden dim
    #           'K': [2, 4, 6],
    #           'multiple_phi': [False, True]}
    #
    # parameter_grid = itertools.product(*params.values())

    # for params_i, parameter_set in enumerate(parameter_grid):
    #     latent_dim, hidden_dim, K, multiple_phi = parameter_set
    #     print(f'Parameter run: Latent dim: {latent_dim}, Hidden dim: {hidden_dim}, K: {K}, Multiple Phis: {multiple_phi}')
    #     start = time.time()  # measure time

    model = GNS(latent_dim=latent_dim, hidden_dim=hidden_dim, K=K, gamma=gamma, multiple_phi=multiple_phi)
    # model.load_state_dict(torch.load(f'../models/Finished models/best_model_c14_K6_L10_H10_True_optimAdam.pth'))

    # # comment CUDA code below for CPU usage
    # if torch.cuda.is_available():
    #     torch.set_default_device('cuda')
    #     model.to('cuda')

    epochs = 10**2+1  # 10**6 used in paper
    lr = 0.001
    # optimizer_name = 'Adagrad'
    optimizer_name = "Adam"
    if optimizer_name == 'Adagrad':
        lr = 0.01
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)  # 0.01 is default for Adagrad
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # warmup_steps = 24
    # lambda1 = lambda step: (lr ** (1 - step / warmup_steps)) if step < warmup_steps else 1
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    best_loss = torch.tensor(float('inf'))
    loss_increase_counter = 0
    case_nr = 14  # 14, 30, 118, 300
    batch_size = 2**7  # 2**7==128
    print_every = 1
    nr_samples = 2**8  # 2**10==1024
    all_buses, all_lines, all_generators = load_all_grids(case_nr, nr_samples=nr_samples)

    wandb_run = wandb.init(
        project="GNS",
        name=f"K{K}, L{latent_dim}, H{hidden_dim}, mul phi: {multiple_phi}",
        config={
            "learning_rate": lr,
            "optimizer": optimizer_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "nr_samples": nr_samples,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "case_nr": case_nr,
            "K": K,
            "nr_samples": nr_samples,
            "gamma": gamma,
            "Multiple Phis": model.multiple_phis})

    for epoch in range(epochs):
        epoch_final_losses = torch.zeros(nr_samples // batch_size)
        for batch_idx_start in range(0, nr_samples, batch_size):
            losses = torch.zeros(batch_size)
            last_losses = torch.zeros(batch_size)
            for i in range(batch_idx_start, batch_idx_start + batch_size):
                buses, lines, generators = all_buses[i], all_lines[i], all_generators[i]
                v, theta, loss, last_loss = model(buses=buses, lines=lines, generators=generators, B=B, L=L, G=G)
                losses[i % batch_size] = loss
                last_losses[i % batch_size] = last_loss.data
            total_loss = torch.mean(losses)
            epoch_final_losses[batch_idx_start // batch_size] = torch.mean(last_losses)

            # print(f'Run: {epoch}, Batch Loss: {total_loss}')
            total_loss.backward()
            optimizer.step()
            # scheduler.step()  # Update the learning rate
            optimizer.zero_grad()

        epoch_final_loss = torch.mean(epoch_final_losses)
        wandb.log({"Final Loss": epoch_final_loss})

        if epoch_final_loss >= best_loss:
            loss_increase_counter += 1
            if loss_increase_counter > 2:
                print('Loss is increasing')
                break
        else:
            best_loss = epoch_final_loss
            best_model = model
            loss_increase_counter = 0

        if epoch % print_every == 0:
            print(f'Epoch: {epoch}, Final Loss: {epoch_final_loss}, best loss: {best_loss}')
            torch.save(best_model.state_dict(),
                       f'../models/best_model_c{case_nr}_K{K}_L{latent_dim}_H{hidden_dim}_{multiple_phi}_optim{optimizer_name}.pth')
        # if epoch == 100:  # epoch starts at 0
        #     # make model a instance of wandb.Artifact
        #     best_model = wandb.Artifact(f'best_model_c{case_nr}_K{K}_L{latent_dim}_H{hidden_dim}_{multiple_phi}_L{torch.ceil(epoch_final_loss)}.pth', type='model')
        #     wandb.log_artifact(best_model)
        # save as model
    wandb_run.finish()
    end = time.time()
    with open(f'time_logs.txt', 'a') as f:
        f.write(f'Latent dim: {latent_dim}, Hidden dim: {hidden_dim}, K: {K}, Multiple Phis: {multiple_phi}, Time: {end-start}\n')


if __name__ == '__main__':
    main()

