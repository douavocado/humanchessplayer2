# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 11:11:03 2023

@author: xusem
"""
import numpy as np
import h5py
from pathlib import Path
import torch
from torch.utils import data

class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, file_path, input_channels=14, recursive=False, load_data=True, data_cache_size=3, transform=None):
        super().__init__()
        self.data = []
        self.labels = []
        self.transform = transform
        self.input_channels=input_channels

        # Search for all h5 files
        p = Path(file_path)
        assert(p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
        self.data = np.vstack(self.data)
        self.labels = np.vstack(self.labels)
            
    def __getitem__(self, index):
        # get data
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x).permute(2,0,1).float()

        # get label
        y = self.labels[index]
        y = torch.from_numpy(y).float()
        return (x, y)

    def __len__(self):
        return len(self.data)
    
    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path) as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    if dname != 'block0_values':
                        continue
                    if gname == 'data':
                        data = ds[()].reshape(ds[()].shape[0],8,8,self.input_channels)[:]
                        self.data.append(data)
                    else:
                        data = ds[()][:]
                        self.labels.append(data)
           
class MoveSelectorDataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, file_path, recursive=False, load_data=True, data_cache_size=3, transform=None):
        super().__init__()
        self.data = []
        self.labels = []
        self.transform = transform

        # Search for all h5 files
        p = Path(file_path)
        assert(p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
        self.data = np.vstack(self.data)
        self.labels = np.vstack(self.labels)
            
    def __getitem__(self, index):
        # get data
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x).float()

        # get label
        y = self.labels[index]
        y = torch.from_numpy(y).float()
        return (x, y)

    def __len__(self):
        return len(self.data)
    
    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path) as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    if dname != 'block0_values':
                        continue
                    if gname == 'data':
                        data = ds[()][:]
                        self.data.append(data)
                    else:
                        data = ds[()][:]
                        self.labels.append(data)            

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct           
   