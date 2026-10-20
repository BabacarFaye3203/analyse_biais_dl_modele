import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CustomCelebADataset(Dataset):
    def __init__(self, txt_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        with open(txt_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        if len(lines[0].split()) == 1:
            self.attr_names = lines[1].split()
            self.data_lines = lines[2:]
        else:
            self.attr_names = lines[0].split()
            self.data_lines = lines[1:]

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        parts = self.data_lines[idx].split()
        img_name = parts[0]
        # Conversion des attributs CelebA de -1/1 vers 0/1
        attributes = [1 if int(x) == 1 else 0 for x in parts[1:]]
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        target_tensor = torch.tensor(attributes, dtype=torch.long)
        return image, target_tensor


def get_default_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])