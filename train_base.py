import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import numpy as np

# Dataset synthétique simulant CelebA avec un biais intentionnel entre l'attribut sensible et la cible
class SyntheticBiasedDataset(Dataset):
    def __init__(self, num_samples=1500):
        # Simulation d'images (tensors 3x224x224)
        self.images = torch.randn(num_samples, 3, 224, 224)
        
        # Attribut sensible (ex: Genre : 0 ou 1)
        self.sensitive = torch.randint(0, 2, (num_samples,))
        
        # Attribut cible (ex: Sourire) avec un BIAIS artificiel fort :
        # Si sensitive == 0, forte probabilité d'avoir target = 0
        # Si sensitive == 1, forte probabilité d'avoir target = 1
        self.targets = torch.zeros(num_samples, dtype=torch.long)
        for i in range(num_samples):
            if self.sensitive[i] == 1:
                self.targets[i] = 1 if np.random.rand() < 0.85 else 0
            else:
                self.targets[i] = 0 if np.random.rand() < 0.85 else 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # On simule le tenseur 'attr' de CelebA (où l'index 31 est la cible 'Smiling' et l'index 20 est l'attribut sensible 'Male')
        attrs = torch.zeros(40, dtype=torch.long)
        attrs[31] = self.targets[idx]
        attrs[20] = self.sensitive[idx]
        return self.images[idx], attrs

def main():
    print("Création du dataset synthétique biaisé (simulation CelebA)...")
    train_dataset = SyntheticBiasedDataset(num_samples=1500)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print("Initialisation de ResNet-50 pré-entraîné...")
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    # Adaptation de la dernière couche pour une classification binaire (2 classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Fonction de perte et optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print(f"Début de l'entraînement sur {device}...")
    model.train()
    for epoch in range(3):  # 3 époques pour un apprentissage stable
        running_loss = 0.0
        for i, (images, attrs) in enumerate(train_loader):
            images = images.to(device)
            targets = attrs[:, 31].to(device)  # Index 31 comme cible

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        print(f"[Époque {epoch+1}/3] Perte moyenne (Loss): {running_loss / len(train_loader):.3f}")

    # Sauvegarde du modèle de base
    torch.save(model.state_dict(), "resnet50_celeba_base.pth")
    print("Modèle de base entraîné et sauvegardé sous 'resnet50_celeba_base.pth' avec succès !")

if __name__ == "__main__":
    main()