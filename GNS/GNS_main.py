import torch
import torch.nn as nn
import random
from pypower.api import case14, case30, case118, case300, ppoption, runpf, printpf
import pickle as pkl
from torch_geometric.data import Data

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
    case_augmented =  pkl.load(open(f'./data/case{case_nr}/augmented_case{case_nr}_{augmentation_nr}.pkl', 'rb'))
    bus_data = torch.tensor(case_augmented['bus'], dtype=torch.float32)
    buses = torch.tensor(bus_data[:, [0, 1, 2, 3, 4, 5]], dtype=torch.float32)
    # Gs and Bs have defaults of 1 in paper, but 0 in matpower
    # Bs is not everywhere 0, but in paper it is everwhere 1 p.u. (of the Qd?)
    buses[:, 4] = buses[:, 3]
    buses[:, 5] = buses[:, 3]
    baseMV = 100  # set to 100 in github, no default in Matpower
    buses[:, 4] /= baseMV  # normalize Gs and Bs to gs and bs by dividing by baseMV
    buses[:, 5] /= baseMV
    buses = torch.cat((buses, torch.zeros((buses.shape[0], 1), dtype=torch.float32)), dim=1)  # add qg column for inserting values
 
    lines_data = torch.tensor(case_augmented['branch'], dtype=torch.float32)
    lines = torch.tensor(lines_data[:, [0, 1, 2, 3, 4, 8, 9]], dtype=torch.float32)
    lines[:, L['tau']] = torch.where(lines[:, L['tau']] == 0, 1, lines[:, L['tau']])

    gen_data = torch.tensor(case_augmented['gen'], dtype=torch.float32)
    generators = torch.tensor(gen_data[:, [0, 8, 9, 1, 5, 2]], dtype=torch.float32)
    generators = torch.cat((generators, generators[:, 3].unsqueeze(dim=1)), dim=1)  # add changable Pg and leave original Pg as Pg_set
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

def global_active_compensation(v, theta, buses, lines, gens, B, L, G):
    p_joule = 0
    # TODO: change p_joule computation such that it is a torch.scatter_add_() operation
    for line_ij in lines:
        i, j = line_ij[:2].int() -1
        y_ij = 1 / torch.sqrt(line_ij[L['r']]**2 + line_ij[L['x']]**2)
        delta_ij = torch.atan2(line_ij[L['r']], line_ij[L['x']])
        p_joule = p_joule + torch.abs(v[i]*v[j]*y_ij/line_ij[L['tau']]*(torch.sin(theta[i] - theta[j] - delta_ij - line_ij[L['theta']]) + torch.sin(theta[j] - theta[i] - delta_ij + line_ij[L['theta']])) + (v[i]/line_ij[L['tau']])**2 * y_ij*torch.sin(delta_ij)+v[j]**2 * y_ij*torch.sin(delta_ij))

    pd_sum = torch.sum(buses[:, B['Pd']])
    gs_sum = torch.sum(buses[:, B['Gs']])
    # original formula below but gs_sum is always zero and somehow it doesnt sum correctly
    pg_sum = gens[:, G['Pg_set']].sum()
    # TODO: change p_global computation such that it is a torch.scatter_add_() operation
    p_global = torch.sum(v.pow(2)) * gs_sum + pd_sum + p_joule 
    # p_global = pd_sum + p_joule

    if p_global < gens[:, G['Pg_set']].sum():
        lambda_ = (p_global - gens[:, G['Pmin']].sum()) / (2*(gens[:, G['Pg_set']].sum() - gens[:, G['Pmin']].sum()))
    else:
        lambda_ = (p_global - 2*gens[:, G['Pg_set']].sum() + gens[:, G['Pmax']].sum()) / (2 * (gens[:, G['Pmax']].sum() - gens[:, G['Pg_set']].sum()))
    # if lambda_ > 1:
    #     lambda_ = 1

    if lambda_ < 0.5:  # equasion (21) in paper
        Pg_new = gens[:, G['Pmin']] + 2*(gens[:, G['Pg_set']] - gens[:, G['Pmin']])*lambda_
    else:
        Pg_new = 2*gens[:, G['Pg_set']] - gens[:, G['Pmax']] + 2*(gens[:, G['Pmax']] - gens[:, G['Pg_set']])*lambda_

    # if Pg is larger than Pmax in any value of the same index, this should be impossible!
    rnd_o1 = torch.rand(1)
    if rnd_o1 < 0.2:
        # if torch.any(Pg_new > gens[:, G['Pmax']]):
        print(f'lambda: {lambda_}')
    qg_new_start = buses[:, B['Qd']] - buses[:, B['Bs']] * v**2
    ## qg_new = [torch.tensor(0, dtype=torch.float32) for _ in range(lines.shape[0])]
    ## find a way to store each new loop variable in a new tensor, then add them all together
    # TODO: change qg_new computation such that it is a torch.scatter_add_() operation
    qg_new = [0. for _ in range(lines.shape[0])]
    for iteration, line_ij in enumerate(lines):
        i, j = line_ij[:2].int() - 1  # -1 as bus indices start from 1
        y_ij = 1 / torch.sqrt(line_ij[L['r']]**2 + line_ij[L['x']]**2)
        delta_ij = torch.atan2(line_ij[L['r']], line_ij[L['x']])
        
        qg_new[iteration] = -torch.sum(-v[i]*v[j]*y_ij/line_ij[L['tau']]*torch.cos(theta[i] - theta[j] - delta_ij - line_ij[L['theta']]) + (v[i]/line_ij[L['tau']])**2 * (y_ij*torch.cos(delta_ij) - line_ij[L['b']]/2)) - torch.sum(-v[j]*v[i]*y_ij/line_ij[L['tau']]*torch.cos(theta[j] - theta[i] - delta_ij - line_ij[L['theta']]) + v[j]**2 *y_ij*torch.sin(delta_ij) - line_ij[L['b']]/2)
    
    whole_sum = torch.zeros((buses.shape[0]), dtype=torch.float32) 
    indices = torch.tensor((lines[:, L['f_bus']]).int() - 1, dtype=torch.int64)
    whole_sum = whole_sum.index_add(0, indices, torch.stack(qg_new))

    qg_new_old = qg_new_start + whole_sum

    return lambda_, Pg_new, qg_new_old

def local_power_imbalance(v, theta, buses, lines, gens, B, L, G):
    delta_p_base = -buses[:, B['Pd']] - buses[:, B['Gs']] * v**2
    # delta_p_zeros = []
    # for i in range(buses.shape[0]):
    #     if i in gens[:, G['bus_i']].int()-1:
    #         # get the value of Pg at the index of the bus, considering gens is smaller than buses 
    #         delta_p_zeros.append(gens[:, G['Pg_set']][gens[:, G['bus_i']].int() - 1 == i])
    #     else:
    #         delta_p_zeros.append(0.)  # changing only at gen indicesg only at gen indices
    delta_p_gens = [gens[:, G['Pg']][gens[:, G['bus_i']].int() - 1 == i] if i in gens[:, G['bus_i']].int()-1 else 0. for i in range(buses.shape[0])]
    delta_p_start = delta_p_base + torch.tensor(delta_p_gens)
    delta_q_start = buses[:, B['qg']] - buses[:, B['Qd']] - buses[:, B['Bs']] * v**2

    delta_p_list = [torch.zeros(1) for _ in range(buses.shape[0])]
    delta_q_list = [torch.zeros(1) for _ in range(buses.shape[0])]
    # TODO: change delta_p and delta_q computation such that it is a torch.scatter_add_() operation
    for line_ij in lines:
        i, j = line_ij[:2].int() - 1  # -1 as bus indices start from 1
        y_ij = 1 / torch.sqrt(line_ij[L['r']]**2 + line_ij[L['x']]**2)
        delta_ij = torch.atan2(line_ij[L['r']], line_ij[L['x']])

        delta_p_list[i] = delta_p_list[i] + torch.sum(v[i]*v[j]*y_ij/line_ij[L['tau']]*(torch.sin(theta[i] - theta[j] - delta_ij - line_ij[L['theta']])) + (v[i]/line_ij[L['tau']])**2 * y_ij*torch.sin(delta_ij)) + torch.sum(v[j]*v[i]*y_ij/line_ij[L['tau']]*(torch.sin(theta[j] - theta[i] - delta_ij - line_ij[L['theta']])) + (v[j]/line_ij[L['tau']])**2 * y_ij*torch.sin(delta_ij))

        delta_q_list[i] = delta_q_list[i] + torch.sum(-v[i]*v[j]*y_ij/line_ij[L['tau']]*torch.cos(theta[i] - theta[j] - delta_ij - line_ij[L['theta']]) + (v[i]/line_ij[L['tau']])**2 * (y_ij*torch.cos(delta_ij) - line_ij[L['b']]/2)) + torch.sum(-v[j]*v[i]*y_ij/line_ij[L['tau']]*torch.cos(theta[j] - theta[i] - delta_ij - line_ij[L['theta']]) + v[j]**2 *y_ij*torch.cos(delta_ij) - line_ij[L['b']]/2)

    delta_p = delta_p_start + torch.tensor(delta_p_list)
    delta_q = delta_q_start + torch.tensor(delta_q_list)

    return delta_p, delta_q


class GNS(nn.Module):
    def __init__(self, latent_dim=10, hidden_dim=10, K=30, gamma=0.9):
        super(GNS, self).__init__()
        # self.correction_block = nn.ModuleDict()

        # self.phi_from = nn.ModuleDict()
        # self.phi_to = nn.ModuleDict()
        # self.phi_loop = nn.ModuleDict()
        # TODO: change architecture of L_theta, L_v, L_m such that they are a torch.geometric GCN layers with K message passes. But still make the global active compensation and local power imbalance functions inside every k pass
        self.phi = nn.ModuleDict()
        self.L_theta = nn.ModuleDict()
        self.L_v = nn.ModuleDict()
        self.L_m = nn.ModuleDict()

        for k in range(K):
            # self.phi_from[str(k)] = LearningBlock(latent_dim + 5, hidden_dim, 1)
            # self.phi_to[str(k)] = LearningBlock(latent_dim + 5, hidden_dim, 1)
            # self.phi_loop[str(k)] = LearningBlock(latent_dim + 5, hidden_dim, 1)
            self.phi[str(k)] = LearningBlock(latent_dim + 5, hidden_dim, 1)
            # self.correction_block[str(k)] = LearningBlock(latent_dim + 5, hidden_dim, 1)
            self.L_theta[str(k)] = LearningBlock(dim_in=5+latent_dim, hidden_dim=hidden_dim, dim_out=1)
            self.L_v[str(k)] = LearningBlock(dim_in=5+latent_dim, hidden_dim=hidden_dim, dim_out=1)
            self.L_m[str(k)] = LearningBlock(dim_in=5+latent_dim, hidden_dim=hidden_dim, dim_out=latent_dim)
        self.gamma = gamma
        self.K = K

    def forward(self, buses, lines, generators, B, L, G):
        m = torch.zeros((buses.shape[0], latent_dim), dtype=torch.float32)
        v = torch.ones((buses.shape[0]), dtype=torch.float32)
        theta = torch.zeros((buses.shape[0]), dtype=torch.float32)
        delta_p = torch.zeros((generators.shape[0]), dtype=torch.float32)
        delta_q = torch.zeros((generators.shape[0]), dtype=torch.float32)
        total_loss = 0.

        v[generators[:, G['bus_i']].long() - 1] = generators[:, G['vg']]
        delta_p = -buses[:, B['Pd']] - buses[:, B['Gs']] * v.pow(2)
        delta_p[generators[:, G['bus_i']].long() - 1] = delta_p[generators[:, G['bus_i']].long() - 1] + generators[:, G['Pg']]
        delta_q = buses[:, B['qg']] - buses[:, B['Qd']] - buses[:, B['Bs']] * v.pow(2)

        for k in range(self.K):
            # TODO: change phi_sum computation such that it is a torch.scatter_add_() aggregation operation
            phi_sum_list = [torch.zeros(1) for _ in range(buses.shape[0])]
            for line_ij in lines:
                i, j = line_ij[:2].long() - 1
                phi_input = torch.cat((m[j], line_ij[2:]), dim=0).unsqueeze(0)
                phi_res = self.phi[str(k)](phi_input).squeeze()
                phi_sum_list[i] = phi_sum_list[i] + phi_res
            phi_sum = torch.stack(phi_sum_list).squeeze()

            network_input = torch.cat((v.unsqueeze(1), theta.unsqueeze(1), delta_p.unsqueeze(1), delta_q.unsqueeze(1), m, phi_sum.unsqueeze(1)), dim=1)
            
            theta_update = self.L_theta[str(k)](network_input)
            theta = theta + theta_update.squeeze()
            
            v_update = self.L_v[str(k)](network_input)
            non_gens_mask = torch.ones_like(v, dtype=torch.bool)
            non_gens_mask[generators[:, G['bus_i']].long() - 1] = False
            v = torch.where(non_gens_mask, v + v_update.squeeze(), v)
            
            m_update = self.L_m[str(k)](network_input)
            m = m + m_update
            
            lambda_, Pg_new, qg_new = global_active_compensation(v, theta, buses, lines, generators, B, L, G)
            generators = torch.cat((generators[:, :G['Pg']], Pg_new.unsqueeze(dim=1)), dim=1)  # no matrix afterwards as Pg is last column
            buses = torch.cat((buses[:, :B['qg']], qg_new.unsqueeze(dim=1), buses[:, B['qg']+1:]), dim=1)
            delta_p, delta_q = local_power_imbalance(v, theta, buses, lines, generators, B, L, G)

            total_loss = total_loss + self.gamma**(self.K-k) * (torch.sum(delta_p.pow(2) + delta_q.pow(2)) / buses.shape[0])

        return v, theta, total_loss


#initialization
latent_dim = 10  # increase later
hidden_dim = 10  # increase later
gamma = 0.9
K = 8  # correction updates, 30 in paper, less for debugging
if torch.cuda.is_available():
    torch.set_default_device('cuda')
model = GNS(latent_dim=latent_dim, hidden_dim=hidden_dim, K=K)
torch.autograd.set_detect_anomaly(True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

n_runs = 10**6  # 10**6 used in paper
best_loss = torch.tensor(float('inf'))
best_model = model
loss_increase_counter = 0
print_every = 1
case_nr = 14  # 14, 30, 118, 300
for run in range(n_runs):
    # sample from different grids
    # augmentation_nr = random.randint(0, 9)  # random augmentation of the 10
    augmentation_nr = 0
    buses, lines, generators = prepare_grid(case_nr, augmentation_nr)

    v, theta, loss = model(buses=buses, lines=lines, generators=generators, B=B, L=L, G=G)
    
    # delta_p, delta_q = local_power_imbalance(v=v, theta=theta, buses=buses, lines=lines, gens=generators, B=B, L=L, G=G)
    
    # last_loss = torch.sum(delta_p**2 + delta_q**2) / buses.shape[0]  # equasion (23)

    # total_loss = torch.sum(loss)
    # total_loss = model.k_loss
    # total_loss = calculate_total_loss(delta_p, delta_q, gamma, K)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if loss >= best_loss:
        loss_increase_counter += 1
        if loss_increase_counter > 50:
            print('Loss is increasing')
            break
    else:
        best_loss = loss.data
        best_model = model
        loss_increase_counter = 0
        torch.save(best_model.state_dict(), f'models/best_model_c{case_nr}_K{K}_L{latent_dim}_H{hidden_dim}_I{run}.pth')
    if run % print_every == 0:
        print(f'Run: {run}, Loss: {loss}, best loss: {best_loss}')
    

