# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 18:41:53 2023

@author: xusem
"""

import torch
from torch.utils.data import DataLoader

from models.train_tools import HDF5Dataset, train_loop, test_loop
from models.models import PieceSelectorNN




train_data_file = 'models/data/piece_to/train/opening'
test_data_file = 'models/data/piece_to/test/opening'

training_dataset = HDF5Dataset(train_data_file, input_channels=16)
testing_dataset = HDF5Dataset(test_data_file, input_channels=16)

train_dataloader = DataLoader(training_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(testing_dataset, batch_size=128, shuffle=True)

model = PieceSelectorNN(input_channels=16)
weights = torch.load('models/model_weights/piece_to_weights_opening.pth', weights_only=True)
model.load_state_dict(weights)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

loss_fn = torch.nn.CrossEntropyLoss()

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
torch.save(model.state_dict(), 'models/model_weights/piece_to_weights_opening.pth')