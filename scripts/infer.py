import torch
from PIL import Image
from torchvision import transforms
from model import TinyUNet
from config import Config

def main(input_path, output_path='output.png'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TinyUNet(Config.in_ch, Config.out_ch, [Config.base_ch * (2**i) for i in range(Config.num_down)])
    model.load_state_dict(torch.load(Config.save_path, map_location=device))
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor()
    ])
    image = transform(Image.open(input_path).convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)

    out_img = transforms.ToPILImage()(output.squeeze().cpu().clamp(0, 1))
    out_img.save(output_path)
    print(f"Saved output image to {output_path}")

if __name__ == '__main__':
    import sys
    main(sys.argv[1])
