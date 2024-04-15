import torch
import pickle as pkl

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


def load_all_grids(case_nr, nr_samples=100, test_set=False):
    all_buses = []
    all_lines = []
    all_generators = []
    start_idx = 1  # i==0 is not augmented case, exclude
    if test_set:
        start_idx = 1000-samples
    for i in range(start_idx, nr_samples+start_idx):
        buses, lines, generators = prepare_grid(case_nr, i)
        all_buses.append(buses)
        all_lines.append(lines)
        all_generators.append(generators)
    return all_buses, all_lines, all_generators
