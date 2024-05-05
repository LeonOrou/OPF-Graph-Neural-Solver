# Graph Neural Solver
This work is based on the paper "Neural Networks for Power Flow: Graph Neural Solver" by Donon, Balthazar, et al.
The goal of this work is to predicting v and theta at each bus of a MATPOWER power grid, scientific paper in process.

Use
> pip install -r requirements.txt

for copying the environment libraries.

## Usage:
(Using pypower for case importation and running comparison algorithms)

Either take data from /data/ or run `augment_grids.py` and replace case_nr with the one desired (9, 14, 30, 118, or 300).

For running your own model, import the `GNS` class and initialize `model()` with the hyperparameters desired and `torch.load_state_dict(PATH_TO_MODEL)`.
The best performing hyperparameters are: K=4, latent_dim=20, hidden_dim=10, multiple_phi=True, gamma=0.9. You may also adapt the learning rate, batch_size, nr_samples, epochs, optimizer, or the learning rate scheduler inside `main.py`. 

For controlling the training, use your own "Weights and Biases" login key and change the config as wanted.

For evaluation, choose a desired case and test size and initialize model as stated above. Just run the script afterwards.



