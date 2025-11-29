# import torch
# import torch.nn as nn
# import numpy as np

# from skimage import measure, morphology, filters
# from scipy.ndimage import gaussian_filter


# # ---------- U-Net definition (same as in train.py) ----------

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


# # ---------- load trained colorization model once ----------

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = UNet().to(device)
# model.load_state_dict(torch.load("unet_colorsar_all_terrains.pth", map_location=device))
# model.eval()


# # ---------- helper: colorize a SAR .npy file ----------

# def colorize_npy(sar_path):
#     """
#     Input: path to SAR .npy file
#     Output: colorized image as numpy array (H, W, 3), values in [0,1]
#     """
#     sar = np.load(sar_path)                       # (H, W)
#     return colorize_array(sar)


# # ---------- helper: colorize a SAR array (for uploaded images) ----------

# def colorize_array(sar_array):
#     """
#     Input: SAR image as numpy array (H, W) in [0,1]
#     Output: colorized image (H, W, 3) in [0,1]
#     """
#     sar_tensor = torch.from_numpy(sar_array).unsqueeze(0).unsqueeze(0).float().to(device)
#     with torch.no_grad():
#         pred = model(sar_tensor)                  # (1, 3, H, W)
#     pred = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
#     return pred


# # ---------- helper: simple SAR-level change overlay (optional) ----------

# def change_overlay(sar1, sar2, thresh=0.2):
#     """
#     Input: two SAR arrays (H, W) in [0,1]
#     Output: RGB overlay (H, W, 3) where change boundaries are red.
#     """
#     diff = np.abs(sar2 - sar1)
#     change_mask = diff > thresh
#     contours = measure.find_contours(change_mask.astype(float), 0.5)

#     base = np.stack([sar2, sar2, sar2], axis=-1)  # gray RGB base from second image

#     for c in contours:
#         c = c.astype(int)
#         base[c[:, 0], c[:, 1]] = [1.0, 0.0, 0.0]  # red boundary

#     return base






# # ---------- helper: change detection on two COLOR images ----------

# def detect_changes_color(color1, color2, min_size=800):
#     """
#     Input: two color images (H, W, 3) in [0,1]
#     Output: overlay image (H, W, 3) with red boundaries on major land-cover changes.
#     """

#     # 1) blur to reduce small texture noise
#     blur1 = gaussian_filter(color1, sigma=1.0)
#     blur2 = gaussian_filter(color2, sigma=1.0)

#     # 2) stronger difference measure:
#     #    L2 magnitude across RGB channels (captures big color/texture shifts)
#     diff = np.abs(blur2 - blur1)                  # (H, W, 3)
#     diff_mag = np.sqrt((diff ** 2).sum(axis=-1))  # (H, W)

#     # normalize to [0,1] for stable thresholding
#     max_val = diff_mag.max()
#     if max_val > 0:
#         diff_mag = diff_mag / max_val

#     # 3) threshold: use Otsu, but slightly LOWER to catch more changes
#     otsu = filters.threshold_otsu(diff_mag)
#     mask = diff_mag > (otsu * 0.9)   # more sensitive than 1.2

#     # 4) morphology: keep medium/large connected regions and merge nearby areas
#     mask = morphology.remove_small_objects(mask, min_size=min_size)      # 800 pixels
#     mask = morphology.remove_small_holes(mask, area_threshold=min_size)
#     mask = morphology.binary_closing(mask, morphology.disk(7))           # stronger merge
#     mask = morphology.binary_opening(mask, morphology.disk(5))           # smooth edges

#     # 5) extract boundaries and draw on second image
#     contours = measure.find_contours(mask.astype(float), 0.5)

#     overlay = color2.copy()
#     for c in contours:
#         c = c.astype(int)
#         overlay[c[:, 0], c[:, 1]] = [1.0, 0.0, 0.0]  # red boundary

#     return overlay






import os
import torch
import torch.nn as nn
import numpy as np

from skimage import measure, morphology, filters
from scipy.ndimage import gaussian_filter


# ---------- U-Net definition (same as in train.py) ----------

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


# ---------- terrain-specific model loading ----------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_loaded_models = {}  # cache {terrain: model}


def load_model_for_terrain(terrain: str):
    """
    Load (or reuse) the UNet model for the given terrain.
    terrain in ['barrenland', 'grassland', 'urban', 'agri']
    """
    global _loaded_models

    terrain = (terrain or 'barrenland').lower()
    if terrain not in ['barrenland', 'grassland', 'urban', 'agri']:
        terrain = 'barrenland'

    if terrain in _loaded_models:
        return _loaded_models[terrain]

    model_path = f"unet_{terrain}.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = UNet().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    _loaded_models[terrain] = model
    return model


# ---------- helper: colorize a SAR .npy file ----------

def colorize_npy(sar_path, terrain='barrenland'):
    """
    Input: path to SAR .npy file, terrain string
    Output: colorized image as numpy array (H, W, 3), values in [0,1]
    """
    sar = np.load(sar_path)          # (H, W)
    return colorize_array(sar, terrain)


# ---------- helper: colorize a SAR array (for uploaded images) ----------

def colorize_array(sar_array, terrain='barrenland'):
    """
    Input: SAR image as numpy array (H, W) in [0,1]
           terrain: 'barrenland' | 'grassland' | 'urban' | 'agri'
    Output: colorized image (H, W, 3) in [0,1]
    """
    model = load_model_for_terrain(terrain)

    sar_tensor = torch.from_numpy(sar_array).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred = model(sar_tensor)   # (1, 3, H, W)
    pred = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return pred


# ---------- helper: simple SAR-level change overlay (optional) ----------

def change_overlay(sar1, sar2, thresh=0.2):
    """
    Input: two SAR arrays (H, W) in [0,1]
    Output: RGB overlay (H, W, 3) where change boundaries are red.
    """
    diff = np.abs(sar2 - sar1)
    change_mask = diff > thresh
    contours = measure.find_contours(change_mask.astype(float), 0.5)

    base = np.stack([sar2, sar2, sar2], axis=-1)  # gray RGB base from second image

    for c in contours:
        c = c.astype(int)
        base[c[:, 0], c[:, 1]] = [1.0, 0.0, 0.0]  # red boundary

    return base


# ---------- helper: change detection on two COLOR images ----------

def detect_changes_color(color1, color2, min_size=800):
    """
    Input: two color images (H, W, 3) in [0,1]
    Output: overlay image (H, W, 3) with red boundaries on major land-cover changes.
    """

    # 1) blur to reduce small texture noise
    blur1 = gaussian_filter(color1, sigma=1.0)
    blur2 = gaussian_filter(color2, sigma=1.0)

    # 2) stronger difference measure:
    #    L2 magnitude across RGB channels (captures big color/texture shifts)
    diff = np.abs(blur2 - blur1)                 # (H, W, 3)
    diff_mag = np.sqrt((diff ** 2).sum(axis=-1)) # (H, W)

    # normalize to [0,1] for stable thresholding
    max_val = diff_mag.max()
    if max_val > 0:
        diff_mag = diff_mag / max_val

    # 3) threshold: use Otsu, but slightly LOWER to catch more changes
    otsu = filters.threshold_otsu(diff_mag)
    mask = diff_mag > (otsu * 0.9)   # more sensitive

    # 4) morphology: keep medium/large connected regions and merge nearby areas
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = morphology.remove_small_holes(mask, area_threshold=min_size)
    mask = morphology.binary_closing(mask, morphology.disk(7))
    mask = morphology.binary_opening(mask, morphology.disk(5))

    # 5) extract boundaries and draw on second image
    contours = measure.find_contours(mask.astype(float), 0.5)

    overlay = color2.copy()
    for c in contours:
        c = c.astype(int)
        overlay[c[:, 0], c[:, 1]] = [1.0, 0.0, 0.0]  # red boundary

    return overlay
