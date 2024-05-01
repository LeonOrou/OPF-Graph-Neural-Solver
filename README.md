# Graph Neural Solver
Optimal Power Flow: Predicting v and theta at each bus, paper in process

> pip install -r requirements.txt

## Usage:
(Using pypower for case importation and running comparison algorithms)\\
Either take data from /data/ or run `augment_grids.py` and replace the case nr with the one desired\\
For running your own model, import the `GNS` class and initialize `model()` with the hyperparameters desired and `torch.load_state_dict(PATH_TO_MODEL)`\\
For evaluation, choose a desired case and test size and initialize model as stated above. Just run the script afterwards\\


