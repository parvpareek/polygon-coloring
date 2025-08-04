class Config:
    data_dir = 'data'
    train_split = 'training'
    val_split = 'validation'
    img_size = 128
    batch_size = 16

    train_json = 'data/training/data.json'
    val_json = 'data/validation/data.json'


    in_ch = 3
    out_ch = 3
    base_ch = 16
    num_down = 2
    emb_dim = 4

    lr = 1e-3
    epochs = 5

    wandb_project = 'polygon-coloring'
    wandb_run_name = 'baseline'
    save_path = 'unet_baseline.pth'
