import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CountEmbedding(nn.Module):
    def __init__(self, dim=64, max_period=10000.0):
        super().__init__()
        half = dim // 2
        self.register_buffer(
            "freq", torch.exp(-math.log(max_period) * torch.arange(half) / half))
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(True))

    def forward(self, r):                      # r ∈ (0,1]
        log_r = torch.log(r).unsqueeze(-1)     # [B,1]
        ang   = log_r * self.freq              # [B,half]
        emb   = torch.cat([torch.sin(ang), torch.cos(ang)], -1)
        return self.proj(emb)                  # [B,dim]

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2) if dropout else nn.Identity()
        )
    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
        )
        self.conv = ConvBlock(in_channels=out_channels*2, out_channels=out_channels, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        diffZ = skip.size(2) - x.size(2)
        diffY = skip.size(3) - x.size(3)
        diffX = skip.size(4) - x.size(4)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2,
                      diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
    
class UNet3D_PDPM(nn.Module):
    def __init__(self, base=32, emb_dim=64):
        super().__init__()
        self.count = CountEmbedding(dim=emb_dim)
        self.enc1, self.p1 = ConvBlock(1, base), nn.MaxPool3d(2)
        self.enc2, self.p2 = ConvBlock(base, base*2), nn.MaxPool3d(2)
        self.enc3, self.p3 = ConvBlock(base*2, base*4), nn.MaxPool3d(2)
        self.enc4, self.p4 = ConvBlock(base*4, base*8), nn.MaxPool3d(2)
        self.bott          = ConvBlock(base*8, base*16, dropout=True)
        self.up4  = UpBlock(base*16, base*8, dropout=True)
        self.up3  = UpBlock(base*8,  base*4, dropout=True)
        self.up2  = UpBlock(base*4,  base*2)
        self.up1  = UpBlock(base*2,  base)
        self.final= nn.Conv3d(base, 1, 1)
        self.proj = nn.Linear(emb_dim, base)

    def forward(self, x, r):
        emb = self.dose(r).view(x.size(0), -1)
        emb = self.proj(emb).view(x.size(0), -1, 1, 1, 1)

        e1 = self.enc1(x) + emb
        e2 = self.enc2(self.p1(e1))
        e3 = self.enc3(self.p2(e2))
        e4 = self.enc4(self.p3(e3))
        b  = self.bott(self.p4(e4))
        d4 = self.up4(b,  e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.final(d1).clamp(0, 1)