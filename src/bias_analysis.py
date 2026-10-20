import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_bias(model, dataloader, target_idx, sensitive_idx, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model.eval()
    all_preds = []
    all_targets = []
    all_sensitives = []
    
    with torch.no_grad():
        for images, attrs in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            targets = attrs[:, target_idx].numpy()
            sensitives = attrs[:, sensitive_idx].numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets)
            all_sensitives.extend(sensitives)
            
    df = pd.DataFrame({
        'target': all_targets,
        'pred': all_preds,
        'sensitive': all_sensitives
    })
    
    metrics = {}
    for group in [0, 1]:
        sub_df = df[df['sensitive'] == group]
        if len(sub_df) > 0:
            acc = accuracy_score(sub_df['target'], sub_df['pred'])
            rec = recall_score(sub_df['target'], sub_df['pred'], zero_division=0)
            f1 = f1_score(sub_df['target'], sub_df['pred'], zero_division=0)
            metrics[group] = {'Accuracy': acc, 'Recall': rec, 'F1-Score': f1, 'Count': len(sub_df)}
            
    # Fairness Discrepancy (Ecart d'exactitude/sensibilité entre groupes)
    if 0 in metrics and 1 in metrics:
        disc_acc = abs(metrics[0]['Accuracy'] - metrics[1]['Accuracy'])
        disc_rec = abs(metrics[0]['Recall'] - metrics[1]['Recall'])
    else:
        disc_acc, disc_rec = 0.0, 0.0
        
    return metrics, disc_acc, disc_rec