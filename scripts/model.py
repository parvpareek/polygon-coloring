import torch
from torch import nn
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1))
    def forward(self, x): return x + self.block(x)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=32):
        super().__init__()
        self.heads, self.dim_head = heads, dim_head
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.to_q, self.to_k, self.to_v = nn.Linear(query_dim, inner_dim, bias=False), nn.Linear(context_dim, inner_dim, bias=False), nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
    def forward(self, x, context):
        B, N, C = x.shape
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        q, k, v = map(lambda t: t.view(B, -1, self.heads, self.dim_head).transpose(1, 2), (q, k, v))
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, -1)
        return self.to_out(out)

class CrossAttnUpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, color_embed_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1), ResBlock(out_ch))
        self.attn = CrossAttention(query_dim=out_ch, context_dim=color_embed_dim)
        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch) 

    def forward(self, x, skip, color_context):
        x = self.up(x); x = torch.cat([x, skip], dim=1); x = self.conv(x)
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        attn_out = self.attn(self.norm1(x_flat), color_context)
        x_flat = self.norm2(x_flat + attn_out) # <-- Apply normalization after residual
        x = x_flat.permute(0, 2, 1).view(B, C, H, W)
        return x

class CrossAttnUNet(nn.Module):
    """A 2-Level version of our stable UNet."""
    def __init__(self, in_ch=3, out_ch=3, base_ch=64, color_embed_dim=64):
        super().__init__()
        # --- 2-Level Encoder ---
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, padding=1), ResBlock(base_ch))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(base_ch, base_ch * 2, 3, padding=1), ResBlock(base_ch * 2))
        self.pool2 = nn.MaxPool2d(2)
        self.bot = ResBlock(base_ch * 2)

        # --- 2-Level Decoder with Cross-Attention ---
        self.dec2 = CrossAttnUpBlock(base_ch * 2, base_ch * 2, base_ch, color_embed_dim)
        self.dec1 = CrossAttnUpBlock(base_ch, base_ch, base_ch, color_embed_dim)
        
        self.final_conv = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x_outline, color_embedding):
        color_context = color_embedding.unsqueeze(1)
        x1 = self.enc1(x_outline); p1 = self.pool1(x1)
        x2 = self.enc2(p1); p2 = self.pool2(x2)
        b = self.bot(p2)
        d2 = self.dec2(b, x2, color_context)
        d1 = self.dec1(d2, x1, color_context)
        return torch.sigmoid(self.final_conv(d1))
