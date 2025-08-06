import os, json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

COLORS_TO_ID = {
    'red': 0, 'green': 1, 'blue': 2, 'yellow': 3,
    'cyan': 4, 'magenta': 5, 'purple': 6, 'orange': 7

}
NUM_COLORS = len(COLORS_TO_ID)

class PolygonDataset(Dataset):
    """
    This dataset version provides:
    1. x_outline: A 3-channel RGB outline image.
    2. y_rgb: The 3-channel ground-truth colored image.
    3. color_id: The numerical ID of the color.
    """
    def __init__(self, root_dir, split, img_size):
        self.img_size = img_size
        base_path = os.path.join(root_dir, split)
        json_path = os.path.join(base_path, 'data.json')
        
        # Using the global COLORS_TO_ID map
        self.COLORS_TO_ID = COLORS_TO_ID

        with open(json_path) as f:
            self.items = json.load(f)
            
        self.base_path = base_path
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        
        input_path = os.path.join(self.base_path, 'inputs', item['input_polygon'])
        output_path = os.path.join(self.base_path, 'outputs', item['output_image'])
        img_in = Image.open(input_path).convert('RGB')
        img_out = Image.open(output_path).convert('RGB')
        
        x_outline_rgb = self.transform(img_in)
        y_rgb = self.transform(img_out)
        
        color_name = item['colour']
        color_id = torch.tensor(self.COLORS_TO_ID.get(color_name, 7), dtype=torch.long)
        
        return x_outline_rgb, y_rgb, color_id

