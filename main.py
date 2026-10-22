import os
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

from src.dataset import CustomCelebADataset, get_default_transforms
from src.model import build_resnet50
from src.train import train_model
from src.bias_analysis import evaluate_bias
from src.mitigator import get_mitigated_dataloader
from src.visualization import plot_bias_comparison


def main():
    txt_path = "./data/list_attr_celeba_500.txt"
    img_dir = "./data/images"
    
    dataset = CustomCelebADataset(txt_file=txt_path, img_dir=img_dir, transform=get_default_transforms())
    
    target_idx = dataset.attr_names.index("Smiling") if "Smiling" in dataset.attr_names else 31
    sensitive_idx = dataset.attr_names.index("Male") if "Male" in dataset.attr_names else 20
    
    print(f"=== Tâche : Cible = '{dataset.attr_names[target_idx]}', Attribut sensible = '{dataset.attr_names[sensitive_idx]}'")
    
    # --- 1. MODÈLE BIAISÉ (ÉTAPE PRÉCÉDENTE) ---
    biased_indices = []
    for idx in range(len(dataset)):
        _, attrs = dataset[idx]
        if attrs[sensitive_idx].item() == 1 and attrs[target_idx].item() == 1:
            if np.random.rand() < 0.10: 
                biased_indices.append(idx)
        else:
            biased_indices.append(idx)
            
    biased_dataset = Subset(dataset, biased_indices)
    dataloader_biased = DataLoader(biased_dataset, batch_size=16, shuffle=True)
    eval_dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    print("\n[1/4] Entraînement du modèle de base (biaisé)...")
    model_base = build_resnet50(num_classes=2)
    model_base = train_model(model_base, dataloader_biased, target_idx, epochs=2, lr=0.0005)
    metrics_base, disc_acc_base, _ = evaluate_bias(model_base, eval_dataloader, target_idx, sensitive_idx)

    # --- 2. MITIGATION DU BIAIS (RÉÉCHANTILLONNAGE PONDÉRÉ) ---
    print("\n[2/4] Application de la stratégie de mitigation (Weighted Sampling)...")
    # On utilise le dataset complet mais avec un rééquilibrage via WeightedRandomSampler
    dataloader_mitigated = get_mitigated_dataloader(dataset, target_idx, sensitive_idx, batch_size=16)

    print("\n[3/4] Entraînement du modèle corrigé (mitigated)...")
    model_mitigated = build_resnet50(num_classes=2)
    model_mitigated = train_model(model_mitigated, dataloader_mitigated, target_idx, epochs=3, lr=0.0005)
    metrics_mitigated, disc_acc_mitigated, _ = evaluate_bias(model_mitigated, eval_dataloader, target_idx, sensitive_idx)

    # --- 3. RÉSULTATS COMPARATIFS ---
    print("\n--- COMPARATIF FINAL ---")
    print(f"  Disparité d'Exactitude (Modèle Biaisé)   : {disc_acc_base:.4f}")
    print(f"  Disparité d'Exactitude (Modèle Corrigé) : {disc_acc_mitigated:.4f}")

    # Sauvegarde des modèles
    os.makedirs("./outputs/models", exist_ok=True)
    torch.save(model_base.state_dict(), "./outputs/models/resnet50_base_biased.pth")
    torch.save(model_mitigated.state_dict(), "./outputs/models/resnet50_mitigated.pth")

    # Génération du graphique comparatif
    print("\n[4/4] Génération des graphiques comparatifs...")
    plot_bias_comparison(metrics_base, metrics_mitigated, save_path="./outputs/figures/bias_comparison.png")
    print("Pipeline de mitigation terminé avec succès !")


if __name__ == "__main__":
    main()