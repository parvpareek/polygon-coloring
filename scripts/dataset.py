import os, json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PolygonColorDataset(Dataset):
    def __init__(self, json_path, img_size=128):
        with open(json_path) as f:
            self.entries = json.load(f)

        base_dir = os.path.dirname(json_path)
        self.input_paths = [os.path.join(base_dir, 'inputs', entry['input']) for entry in self.entries]
        self.output_paths = [os.path.join(base_dir, 'outputs', entry['output']) for entry in self.entries]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        x = Image.open(self.input_paths[idx]).convert('RGB')
        y = Image.open(self.output_paths[idx]).convert('RGB')
        return self.transform(x), self.transform(y)
