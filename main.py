import os
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

from src.dataset import CustomCelebADataset, get_default_transforms
from src.model import build_resnet50
from src.train import train_model
from src.bias_analysis import evaluate_bias


def main():
    txt_path = "./data/list_attr_celeba_500.txt"
    img_dir = "./data/images"
    
    dataset = CustomCelebADataset(txt_file=txt_path, img_dir=img_dir, transform=get_default_transforms())
    
    # Indices dans CelebA : 31 = 'Smiling' (Cible), 20 = 'Male' (Sensible)
    target_idx = dataset.attr_names.index("Smiling") if "Smiling" in dataset.attr_names else 31
    sensitive_idx = dataset.attr_names.index("Male") if "Male" in dataset.attr_names else 20
    
    print(f"=== Tâche : Cible = '{dataset.attr_names[target_idx]}', Attribut sensible = '{dataset.attr_names[sensitive_idx]}' ===")
    
    # --- CRÉATION D'UN BIAIS INTENTIONNEL POUR LE MODÈLE DE BASE ---
    # On filtre le dataset d'entraînement pour enlever une grande partie des hommes qui sourient,
    # ce qui va forcer le modèle à rater ce sous-groupe (biais de stéréotype).
    biased_indices = []
    for idx in range(len(dataset)):
        _, attrs = dataset[idx]
        is_male = attrs[sensitive_idx].item()
        is_smiling = attrs[target_idx].item()
        
        # Si c'est un homme qui sourit, on l'exclut majoritairement de l'entraînement
        if is_male == 1 and is_smiling == 1:
            if np.random.rand() < 0.10: 
                biased_indices.append(idx)
        else:
            biased_indices.append(idx)
            
    biased_dataset = Subset(dataset, biased_indices)
    dataloader_biased = DataLoader(biased_dataset, batch_size=16, shuffle=True)
    
    # Le dataloader complet servira pour l'évaluation objective
    eval_dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    # -------------------------------------------------------------
    # Étape 1 & 2 : Modèle de Base Biaisé & Analyse Initiale
    # -------------------------------------------------------------
    print("\n[1/2] Entraînement du modèle de base sur des données biaisées...")
    model_base = build_resnet50(num_classes=2)
    # Entraînement rapide sur 2 époques avec un learning rate un peu plus fort
    model_base = train_model(model_base, dataloader_biased, target_idx, epochs=2, lr=0.0005)
    
    print("\n[2/2] Évaluation des biais du modèle de base...")
    metrics_base, disc_acc_base, disc_rec_base = evaluate_bias(model_base, eval_dataloader, target_idx, sensitive_idx)
    
    print("\n--- RÉSULTATS DU MODÈLE DE BASE (BIAISÉ) ---")
    print(f"  Performances Groupe 0 (Femmes) : {metrics_base.get(0, {})}")
    print(f"  Performances Groupe 1 (Hommes)  : {metrics_base.get(1, {})}")
    print(f"  --> Disparité d'Exactitude (Accuracy Discrepancy) : {disc_acc_base:.4f}")
    
    # Sauvegarde du modèle de base biaisé
    os.makedirs("./outputs/models", exist_ok=True)
    torch.save(model_base.state_dict(), "./outputs/models/resnet50_base_biased.pth")
    print("\nModèle de base biaisé sauvegardé sous './outputs/models/resnet50_base_biased.pth'.")


if __name__ == "__main__":
    main()