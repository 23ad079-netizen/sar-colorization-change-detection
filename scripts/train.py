# import os
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split

# # 1. LOAD ALL TERRAIN DATA

# data_dir = r'C:\Users\Student\OneDrive\Desktop\color\data\processed\train'

# # Get all SAR and color npy files from all terrains
# sar_files = []
# color_files = []

# for f in os.listdir(data_dir):
#     if f.endswith('_sar.npy'):
#         sar_files.append(os.path.join(data_dir, f))
#     elif f.endswith('_color.npy'):
#         color_files.append(os.path.join(data_dir, f))

# # Sort and pair by common prefix (everything before "_sar.npy" / "_color.npy")
# def base_name(path, suffix):
#     return os.path.basename(path).replace(suffix, '')

# sar_files = sorted(sar_files, key=lambda p: base_name(p, '_sar.npy'))
# color_files = sorted(color_files, key=lambda p: base_name(p, '_color.npy'))

# assert len(sar_files) == len(color_files), f"SAR files: {len(sar_files)}, Color files: {len(color_files)}"

# # Check pairing one example
# print("Example pair:")
# print(sar_files[0])
# print(color_files[0])

# # Train/validation split (80/20)
# train_sar, val_sar, train_col, val_col = train_test_split(
#     sar_files, color_files, test_size=0.2, random_state=42
# )

# print(f"Total pairs: {len(sar_files)}")
# print(f"Train pairs: {len(train_sar)}, Val pairs: {len(val_sar)}")

# # 2. DATASET CLASS

# class SARColorDataset(Dataset):
#     def __init__(self, sar_list, color_list):
#         self.sar_list = sar_list
#         self.color_list = color_list

#     def __len__(self):
#         return len(self.sar_list)

#     def __getitem__(self, idx):
#         sar = np.load(self.sar_list[idx])
#         color = np.load(self.color_list[idx])
#         sar = torch.from_numpy(sar).unsqueeze(0).float()          # (1,H,W)
#         color = torch.from_numpy(color).permute(2,0,1).float()    # (3,H,W)
#         return sar, color

# train_dataset = SARColorDataset(train_sar, train_col)
# val_dataset = SARColorDataset(val_sar, val_col)

# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# # 3. U-NET MODEL (same as before)

# class UNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.enc1 = nn.Sequential(
#             nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
#             nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
#         )
#         self.pool1 = nn.MaxPool2d(2)

#         self.enc2 = nn.Sequential(
#             nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
#         )
#         self.pool2 = nn.MaxPool2d(2)

#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()
#         )

#         self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#         self.dec1 = nn.Sequential(
#             nn.Conv2d(128, 64, 3, padding=1), nn.ReLU()
#         )

#         self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
#         self.dec2 = nn.Sequential(
#             nn.Conv2d(64, 32, 3, padding=1), nn.ReLU()
#         )

#         self.out_conv = nn.Conv2d(32, 3, 1)

#     def forward(self, x):
#         e1 = self.enc1(x)
#         p1 = self.pool1(e1)

#         e2 = self.enc2(p1)
#         p2 = self.pool2(e2)

#         b = self.bottleneck(p2)

#         up1 = self.up1(b)
#         cat1 = torch.cat([up1, e2], dim=1)
#         d1 = self.dec1(cat1)

#         up2 = self.up2(d1)
#         cat2 = torch.cat([up2, e1], dim=1)
#         d2 = self.dec2(cat2)

#         out = self.out_conv(d2)
#         return torch.sigmoid(out)

# # 4. TRAINING SETUP

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = UNet().to(device)
# loss_fn = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# num_epochs = 5   # increase later if training is stable

# # 5. TRAINING LOOP

# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0.0
#     for sar, color in train_loader:
#         sar, color = sar.to(device), color.to(device)
#         optimizer.zero_grad()
#         outputs = model(sar)
#         loss = loss_fn(outputs, color)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     avg_train = total_loss / len(train_loader)

#     # simple validation loss
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for sar, color in val_loader:
#             sar, color = sar.to(device), color.to(device)
#             outputs = model(sar)
#             loss = loss_fn(outputs, color)
#             val_loss += loss.item()
#     avg_val = val_loss / len(val_loader)

#     print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train:.4f}, Val Loss = {avg_val:.4f}")

# torch.save(model.state_dict(), "unet_colorsar_all_terrains.pth")
# print("Training finished and model saved as unet_colorsar_all_terrains.pth!")






import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# 0. PARSE COMMAND LINE ARGUMENTS

parser = argparse.ArgumentParser(description='Train terrain-specific SAR colorization model')
parser.add_argument('--terrain', type=str, default='barrenland', 
                    choices=['barrenland', 'grassland', 'urban', 'agri'],
                    help='Terrain type to train on')
args = parser.parse_args()

terrain = args.terrain
print(f"Training model for terrain: {terrain}")


# 1. LOAD TERRAIN-SPECIFIC DATA (by filename prefix)

data_dir = r'C:\Users\Student\color\data\processed\train'

# Get all SAR and color npy files that START WITH the terrain prefix
sar_files = []
color_files = []

for f in os.listdir(data_dir):
    # Check if filename starts with terrain name (case-insensitive)
    if f.lower().startswith(terrain.lower() + '_'):
        if f.endswith('_sar.npy'):
            sar_files.append(os.path.join(data_dir, f))
        elif f.endswith('_color.npy'):
            color_files.append(os.path.join(data_dir, f))

if len(sar_files) == 0:
    print(f"ERROR: No files found for terrain '{terrain}'")
    print(f"Expected files starting with: '{terrain}_'")
    print(f"Available files: {os.listdir(data_dir)[:5]}")  # show first 5
    exit(1)

# Sort and pair by common prefix
def base_name(path, suffix):
    return os.path.basename(path).replace(suffix, '')

sar_files = sorted(sar_files, key=lambda p: base_name(p, '_sar.npy'))
color_files = sorted(color_files, key=lambda p: base_name(p, '_color.npy'))

assert len(sar_files) == len(color_files), f"SAR files: {len(sar_files)}, Color files: {len(color_files)}"

# Check pairing one example
print("Example pair:")
print(sar_files[0])
print(color_files[0])

# Train/validation split (80/20)
train_sar, val_sar, train_col, val_col = train_test_split(
    sar_files, color_files, test_size=0.2, random_state=42
)

print(f"Total pairs for {terrain}: {len(sar_files)}")
print(f"Train pairs: {len(train_sar)}, Val pairs: {len(val_sar)}")


# 2. DATASET CLASS

class SARColorDataset(Dataset):
    def __init__(self, sar_list, color_list):
        self.sar_list = sar_list
        self.color_list = color_list

    def __len__(self):
        return len(self.sar_list)

    def __getitem__(self, idx):
        sar = np.load(self.sar_list[idx])
        color = np.load(self.color_list[idx])
        sar = torch.from_numpy(sar).unsqueeze(0).float()          # (1,H,W)
        color = torch.from_numpy(color).permute(2,0,1).float()    # (3,H,W)
        return sar, color


train_dataset = SARColorDataset(train_sar, train_col)
val_dataset = SARColorDataset(val_sar, val_col)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


# 3. U-NET MODEL

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


# 4. TRAINING SETUP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

num_epochs = 8


# 5. TRAINING LOOP

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for sar, color in train_loader:
        sar, color = sar.to(device), color.to(device)
        optimizer.zero_grad()
        outputs = model(sar)
        loss = loss_fn(outputs, color)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train = total_loss / len(train_loader)

    # Validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for sar, color in val_loader:
            sar, color = sar.to(device), color.to(device)
            outputs = model(sar)
            loss = loss_fn(outputs, color)
            val_loss += loss.item()
    avg_val = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train:.4f}, Val Loss = {avg_val:.4f}")


# Save model with terrain name
model_save_path = f"unet_{terrain}.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Training finished and model saved as {model_save_path}!")
