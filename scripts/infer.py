import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from config import CONFIG, DEVICE
from dataset import PolygonDataset
from model import CrossAttnUNet

val_ds = PolygonDataset(CONFIG['data_root'], 'validation', CONFIG['img_size'])
val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)

# Build model and embedding
model = CrossAttnUNet(CONFIG['in_ch'], CONFIG['out_ch'], CONFIG['base_ch'], CONFIG['color_embed_dim']).to(DEVICE)
color_emb = torch.nn.Embedding(len(val_ds.COLORS_TO_ID), CONFIG['color_embed_dim']).to(DEVICE)

# Load weights
model.load_state_dict(torch.load('model.pth', map_location=DEVICE))
color_emb.load_state_dict(torch.load('color_emb.pth', map_location=DEVICE))
model.eval(); color_emb.eval()

# Inference & visualize
imgs, gts, cols = next(iter(val_loader))
imgs, gts, cols = imgs.to(DEVICE), gts.to(DEVICE), cols.to(DEVICE)
with torch.no_grad():
    preds = model(imgs, color_emb(cols))

# Plot
fig, axes = plt.subplots(len(imgs), 3, figsize=(9, 3*len(imgs)))
for i in range(len(imgs)):
    axes[i,0].imshow(imgs[i].cpu().permute(1,2,0))
    axes[i,0].set_title('Input')
    axes[i,0].axis('off')
    axes[i,1].imshow(gts[i].cpu().permute(1,2,0))
    axes[i,1].set_title('Ground Truth')
    axes[i,1].axis('off')
    axes[i,2].imshow(preds[i].cpu().permute(1,2,0))
    axes[i,2].set_title('Prediction')
    axes[i,2].axis('off')
plt.tight_layout()
plt.show()