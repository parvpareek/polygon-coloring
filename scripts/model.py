import torch.nn as nn
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, features):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        chs = [in_ch] + features

        for i in range(len(features)):
            self.downs.append(DoubleConv(chs[i], chs[i+1]))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        rev_feats = features[::-1]
        for i in range(len(rev_feats)-1):
            self.ups.append(nn.ConvTranspose2d(rev_feats[i]*2, rev_feats[i+1], 2, 2))
            self.ups.append(DoubleConv(rev_feats[i]*2, rev_feats[i+1]))

        self.final = nn.Conv2d(rev_feats[-1], out_ch, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = nn.MaxPool2d(2)(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            x = torch.cat((x, skips[i//2]), dim=1)
            x = self.ups[i+1](x)

        return self.final(x)
