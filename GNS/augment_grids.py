from pypower.api import case14, case30, case118, case300
import numpy as np
import os
import pickle
import copy

# Load the caseXXX dataset from pypower.api library, change XXX to one of [14, 30, 118, 300]
case = case118()  # change caseXXX to one of [case14(), case30(), case118(), case300()]
case_nr = 118  # change XXX to one of [14, 30, 118, 300]

# Define the augmentation ranges
r_range = [0.9, 1.1]
x_range = [0.9, 1.1]
b_range = [0.9, 1.1]
tau_range = [0.8, 1.2]
theta_shift_range = [-0.2, 0.2]
vg_range = [0.95, 1.05]
pg_range = [0.25, 0.75]
pd_range = [0.5, 1.5]
qd_range = [0.5, 1.5]

num_augmentations = 10000
# Perform data augmentations
augmented_data = []
for _ in range(num_augmentations):
    if len(augmented_data) == 0:
        augmented_data.append(copy.deepcopy(case))
    augmented_case = copy.deepcopy(case)
    # Perturb r, x, b
    augmented_case['branch'][:, 2] *= np.random.uniform(*r_range, size=augmented_case['branch'].shape[0])
    augmented_case['branch'][:, 3] *= np.random.uniform(*x_range, size=augmented_case['branch'].shape[0])
    augmented_case['branch'][:, 4] *= np.random.uniform(*b_range, size=augmented_case['branch'].shape[0])
    # Perturb tau
    augmented_case['branch'][:, 8] = np.random.uniform(*tau_range, size=augmented_case['branch'].shape[0])
    # Perturb theta shift
    augmented_case['branch'][:, 9] = np.random.uniform(*theta_shift_range, size=augmented_case['branch'].shape[0])
    # Perturb vg
    augmented_case['gen'][:, 5] *= np.random.uniform(*vg_range, size=augmented_case['gen'].shape[0])
    # Perturb pg
    pg_max = (augmented_case['gen'][:, 8] - augmented_case['gen'][:, 9]) * pg_range[1]  # in case Pmin is not 0, the range changes
    pg_min = (augmented_case['gen'][:, 8] - augmented_case['gen'][:, 9]) * pg_range[0]
    augmented_case['gen'][:, 1] = np.random.uniform(augmented_case['gen'][:, 9] + pg_min, pg_max, size=augmented_case['gen'].shape[0])
    # Perturb pd
    augmented_case['bus'][:, 2] *= np.random.uniform(*pd_range, size=augmented_case['bus'].shape[0])
    # calculate the difference between all generated and all demanded power as a common factor, multiply all pd with this factor for equality
    augmented_case['bus'][:, 2] *= np.sum(augmented_case['gen'][:, 1]) / np.sum(augmented_case['bus'][:, 2])
    # Perturb qd
    augmented_case['bus'][:, 3] *= np.random.uniform(*qd_range, size=augmented_case['bus'].shape[0])
    augmented_data.append(augmented_case)

# Save augmented grid locally as .pkl file
augmented_data_dir = f'../data/case{case_nr}'
os.makedirs(augmented_data_dir, exist_ok=True)
for i, augmented_case in enumerate(augmented_data):
    with open(f'{augmented_data_dir}/augmented_case{case_nr}_{i}.pkl', 'wb') as f:
        pickle.dump(augmented_case, f)
