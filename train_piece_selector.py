# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 11:19:32 2023

@author: xusem
"""

import torch
from torch.utils.data import DataLoader

from models.train_tools import HDF5Dataset, train_loop, test_loop
from models.models import PieceSelectorNN




train_data_file = 'models/data/defensive_tactics/piece_selector/train/'
test_data_file = 'models/data/defensive_tactics/piece_selector/test/'

training_dataset = HDF5Dataset(train_data_file)
testing_dataset = HDF5Dataset(test_data_file)

train_dataloader = DataLoader(training_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(testing_dataset, batch_size=128, shuffle=True)

model = PieceSelectorNN()
weights = torch.load('models/model_weights/piece_selector_midgame_weights.pth', weights_only=True)
model.load_state_dict(weights)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

loss_fn = torch.nn.CrossEntropyLoss()

epochs = 50
best_loss = float('inf')

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loss, test_acc = test_loop(test_dataloader, model, loss_fn)
    
    # Save model if it has the lowest loss so far
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), 'models/model_weights/piece_selector_defensive_tactics_weights.pth')
        print(f"New best model saved! Loss: {best_loss:.6f}, Accuracy: {test_acc:.4f}")

print("Done!")
print(f"Best model had loss: {best_loss:.6f}")