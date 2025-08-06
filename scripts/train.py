import torch
import lpips
from torch.utils.data import DataLoader
import torchvision
import wandb
from config import CONFIG, DEVICE, ACCELERATOR
from dataset import PolygonDataset, NUM_COLORS
from model import CrossAttnUNet
from torch import nn, optim

# Data
train_ds = PolygonDataset(CONFIG['data_root'], 'training', CONFIG['img_size'])
val_ds   = PolygonDataset(CONFIG['data_root'], 'validation', CONFIG['img_size'])
train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)

# Model, embedding, optimizer, scheduler, loss
model = CrossAttnUNet(CONFIG['in_ch'], CONFIG['out_ch'], CONFIG['base_ch'], CONFIG['color_embed_dim']).to(DEVICE)
color_emb = nn.Embedding(NUM_COLORS, CONFIG['color_embed_dim']).to(DEVICE)
optimizer = optim.Adam(list(model.parameters()) + list(color_emb.parameters()), lr=CONFIG['lr'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)
criterion_l1 = nn.L1Loss()
criterion_lpips = lpips.LPIPS(net='alex').to(DEVICE)

# Accelerator setup
model, color_emb, optimizer, train_loader, val_loader = ACCELERATOR.prepare(
    model, color_emb, optimizer, train_loader, val_loader
)

# W&B init
wandb.login()
wandb.init(project=CONFIG['project'], name=CONFIG['run_name'], config=CONFIG)

# Training loop
for epoch in range(CONFIG['epochs']):
    model.train()
    for x_o, y, c in train_loader:
        x_o, y, c = x_o.to(DEVICE), y.to(DEVICE), c.to(DEVICE)
        color_vec = color_emb(c)
        pred = model(x_o, color_vec)

        loss_l1 = criterion_l1(pred, y)
        loss_lpips = criterion_lpips(pred, y).mean()
        loss = loss_l1 + CONFIG['lpips_weight'] * loss_lpips

        optimizer.zero_grad()
        ACCELERATOR.backward(loss)
        optimizer.step()

    scheduler.step()

    # Validation logging
    model.eval()
    x_o_val, y_val, c_val = next(iter(val_loader))
    x_o_val, y_val, c_val = x_o_val.to(DEVICE), y_val.to(DEVICE), c_val.to(DEVICE)
    with torch.no_grad():
        pred_val = model(x_o_val, color_emb(c_val))
        grid = torchvision.utils.make_grid(
            torch.cat([x_o_val, y_val, pred_val]), nrow=x_o_val.size(0), normalize=True
        )
        wandb.log({"predictions": wandb.Image(grid, caption=f"Epoch {epoch+1}")})

# Save
torch.save(model.state_dict(), "model.pth")
torch.save(color_emb.state_dict(), "color_emb.pth")
wandb.finish()