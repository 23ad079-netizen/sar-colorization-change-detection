import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Define the same UNet used in train.py
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU()
        )

        self.out_conv = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        up1 = self.up1(b)
        cat1 = torch.cat([up1, e2], dim=1)
        d1 = self.dec1(cat1)

        up2 = self.up2(d1)
        cat2 = torch.cat([up2, e1], dim=1)
        d2 = self.dec2(cat2)

        out = self.out_conv(d2)
        return torch.sigmoid(out)

# 2. Load trained model (all terrains)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load("unet_colorsar_all_terrains.pth", map_location=device))
model.eval()

# 3. Path to the SAR .npy you want to colorize
sample_sar_file = r'C:\Users\Student\color\data\processed\train\urban_ROIs1970_fall_s1_144_p414_sar.npy'

# 4. Load and prepare SAR data
sar = np.load(sample_sar_file)          # shape (256,256), values 0–1
sar_tensor = torch.from_numpy(sar).unsqueeze(0).unsqueeze(0).float().to(device)
# shape now (1,1,H,W)

# 5. Run prediction
with torch.no_grad():
    pred_color = model(sar_tensor)      # (1,3,H,W)
    pred_color = pred_color.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # now (H,W,3) for display

# 6. Show and save result
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Input SAR")
plt.imshow(sar, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Predicted Color")
plt.imshow(pred_color)
plt.axis('off')

plt.tight_layout()

# Save into results/colorized folder
out_path = r'C:\Users\Student\color\results\colorized\colorized_urban_example.png'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path)

plt.show()
