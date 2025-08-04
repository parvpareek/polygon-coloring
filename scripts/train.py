import torch
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import L1Loss
from dataset import PolygonColorDataset
from model import UNet
from config import Config

def main():
    wandb.init(
        project=Config.wandb_project,
        name=Config.wandb_run_name,
        config=vars(Config)
    )

    accelerator = Accelerator()

    train_ds = PolygonColorDataset(Config.train_json, Config.img_size)
    val_ds   = PolygonColorDataset(Config.val_json,   Config.img_size)

    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.batch_size)

    model = UNet(Config.in_ch, Config.out_ch, [Config.base_ch * (2**i) for i in range(Config.num_down)])
    optimizer = Adam(model.parameters(), lr=Config.lr)
    loss_fn = L1Loss()

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            pred = model(x)
            loss = loss_fn(pred, y)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                val_loss += loss_fn(pred, y).item()

        wandb.log({
            'epoch': epoch,
            'train_loss': total_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader)
        })

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(model).state_dict(), Config.save_path)
        wandb.save(Config.save_path)

if __name__ == '__main__':
    main()
