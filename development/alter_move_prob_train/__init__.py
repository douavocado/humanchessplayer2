"""Training tooling for AlterMoveProbNN.

The model itself lives in models/alter_move_prob_nn.py and its weights in
models/model_weights/ -- engine.py loads both at startup, so they are not
development code. Only the data generation / training / evaluation scripts
remain here.
"""
