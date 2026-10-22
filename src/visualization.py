import matplotlib.pyplot as plt
import os

def plot_bias_comparison(metrics_base, metrics_mitigated, save_path="./outputs/figures/bias_comparison.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    groups = ['Femmes (Groupe 0)', 'Hommes (Groupe 1)']
    
    # Récupération des métriques
    base_acc = [metrics_base.get(0, {}).get('Accuracy', 0), metrics_base.get(1, {}).get('Accuracy', 0)]
    mit_acc = [metrics_mitigated.get(0, {}).get('Accuracy', 0), metrics_mitigated.get(1, {}).get('Accuracy', 0)]
    
    base_rec = [metrics_base.get(0, {}).get('Recall', 0), metrics_base.get(1, {}).get('Recall', 0)]
    mit_rec = [metrics_mitigated.get(0, {}).get('Recall', 0), metrics_mitigated.get(1, {}).get('Recall', 0)]
    
    x = range(len(groups))
    width = 0.35
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- GRAPHIQUE 1 : ACCURACY ---
    axes[0].bar([i - width/2 for i in x], base_acc, width, label='Modèle Biaisé', color='#e74c3c')
    axes[0].bar([i + width/2 for i in x], mit_acc, width, label='Modèle Corrigé', color='#2ecc71')
    axes[0].set_ylabel('Exactitude (Accuracy)')
    axes[0].set_title('Comparaison de l\'Exactitude par Sous-groupe')
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(groups)
    axes[0].set_ylim(0, 1.1)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # --- GRAPHIQUE 2 : RECALL ---
    axes[1].bar([i - width/2 for i in x], base_rec, width, label='Modèle Biaisé', color='#e74c3c')
    axes[1].bar([i + width/2 for i in x], mit_rec, width, label='Modèle Corrigé', color='#3498db')
    axes[1].set_ylabel('Rappel (Recall)')
    axes[1].set_title('Comparaison du Rappel par Sous-groupe')
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(groups)
    axes[1].set_ylim(0, 1.1)
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Graphique comparatif sauvegardé avec succès dans : {save_path}")
    plt.show()