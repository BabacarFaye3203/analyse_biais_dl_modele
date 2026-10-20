import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, dataloader, target_idx, epochs=3, lr=0.0001, class_weights=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, attrs) in enumerate(dataloader):
            images = images.to(device)
            targets = attrs[:, target_idx].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"  [Époque {epoch+1}/{epochs}] Perte moyenne : {running_loss / len(dataloader):.4f}")
        
    return model