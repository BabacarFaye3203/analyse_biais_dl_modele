import torch
import numpy as np


def compute_class_weights(dataset, target_idx):
    targets = []
    for _, attrs in dataset:
        targets.append(attrs[target_idx].item())
        
    targets = np.array(targets)
    classes, counts = np.unique(targets, return_counts=True)
    
    total = len(targets)
    weights = total / (len(classes) * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)