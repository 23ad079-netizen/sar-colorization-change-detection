import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils import UNet  # reuse architecture


class SARColorDataset(Dataset):
    def __init__(self, sar_list, color_list):
        self.sar_list = sar_list
        self.color_list = color_list

    def __len__(self):
        return len(self.sar_list)

    def __getitem__(self, idx):
        sar = np.load(self.sar_list[idx])
        color = np.load(self.color_list[idx])
        sar = torch.from_numpy(sar).unsqueeze(0).float()       # (1,H,W)
        color = torch.from_numpy(color).permute(2,0,1).float() # (3,H,W)
        return sar, color


def get_file_pairs(data_dir, terrain):
    sar_files, color_files = [], []
    for f in os.listdir(data_dir):
        if f.lower().startswith(terrain.lower() + '_'):
            if f.endswith('_sar.npy'):
                sar_files.append(os.path.join(data_dir, f))
            elif f.endswith('_color.npy'):
                color_files.append(os.path.join(data_dir, f))

    def base_name(path, suffix):
        return os.path.basename(path).replace(suffix, '')

    sar_files = sorted(sar_files, key=lambda p: base_name(p, '_sar.npy'))
    color_files = sorted(color_files, key=lambda p: base_name(p, '_color.npy'))
    assert len(sar_files) == len(color_files)
    return sar_files, color_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--terrain', type=str, required=True,
                        choices=['barrenland', 'grassland', 'urban', 'agri'])
    args = parser.parse_args()
    terrain = args.terrain

    data_dir = r'C:\Users\Student\color\data\processed\train'

    sar_files, color_files = get_file_pairs(data_dir, terrain)
    _, val_sar, _, val_col = train_test_split(
        sar_files, color_files, test_size=0.2, random_state=42
    )

    val_dataset = SARColorDataset(val_sar, val_col)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = f'unet_{terrain}.pth'
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    l1 = nn.L1Loss(reduction='mean')

    total_l1 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0

    with torch.no_grad():
        for sar, gt in val_loader:
            sar = sar.to(device)
            gt = gt.to(device)

            pred = model(sar)                      # (1,3,H,W)
            loss_l1 = l1(pred, gt).item()
            total_l1 += loss_l1

            # convert to numpy in [0,1]
            pred_np = pred.squeeze(0).permute(1,2,0).cpu().numpy()
            gt_np   = gt.squeeze(0).permute(1,2,0).cpu().numpy()

            psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0)
            ssim = structural_similarity(gt_np, pred_np, data_range=1.0, channel_axis=2)

            total_psnr += psnr
            total_ssim += ssim
            n += 1

    print(f"Terrain: {terrain}")
    print(f"Validation samples: {n}")
    print(f"Mean L1 (MAE): {total_l1 / n:.4f}")
    print(f"Mean PSNR (dB): {total_psnr / n:.2f}")
    print(f"Mean SSIM: {total_ssim / n:.4f}")


if __name__ == '__main__':
    main()
