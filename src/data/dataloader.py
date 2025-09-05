import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FramesDataset(Dataset):
    def __init__(self, frames_dir, transform=None):
        self.frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        self.transform = transform or transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        p = self.frames[idx]
        img = Image.open(p).convert('RGB')
        x = self.transform(img)
        # Placeholder: dummy label 0 (user should replace with real labels for training)
        y = 0
        return x, y
