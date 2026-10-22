import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
import numpy as np

def get_mitigated_dataloader(dataset, target_idx, sensitive_idx, batch_size=16):
    """
    Crée un DataLoader pondéré (WeightedRandomSampler) pour rééquilibrer 
    les sous-groupes lors de l'entraînement et corriger le biais.
    """
    targets = []
    class_counts = {}
    
    # Calcul des poids pour chaque échantillon pour donner plus de chances 
    # aux groupes sous-représentés d'être tirés au sort.
    sample_weights = []
    for idx in range(len(dataset)):
        _, attrs = dataset[idx]
        is_male = attrs[sensitive_idx].item()
        is_smiling = attrs[target_idx].item()
        
        # Définition des combinaisons (Genre x Sourire)
        group_key = (is_male, is_smiling)
        class_counts[group_key] = class_counts.get(group_key, 0) + 1
        targets.append(group_key)
        
    # Calcul inverse de la fréquence pour pondérer
    total_samples = len(dataset)
    weights_map = {k: total_samples / count for k, count in class_counts.items()}
    
    for group_key in targets:
        sample_weights.append(weights_map[group_key])
        
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)