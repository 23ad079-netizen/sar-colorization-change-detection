import os
import cv2
import numpy as np

# Absolute paths on your Windows machine
base_path = r'C:\Users\Student\color\data\raw\sentinel12-image-pairs-segregated-by-terrain\v_2'
terrains = ['agri', 'barrenland', 'grassland', 'urban']

output_dir = r'C:\Users\Student\color\data\processed\train'
os.makedirs(output_dir, exist_ok=True)

for terrain in terrains:
    sar_dir = os.path.join(base_path, terrain, 's1')
    color_dir = os.path.join(base_path, terrain, 's2')
    sar_images = sorted(os.listdir(sar_dir))
    color_images = sorted(os.listdir(color_dir))
    for sar_img, color_img in zip(sar_images, color_images):
        sar_path = os.path.join(sar_dir, sar_img)
        color_path = os.path.join(color_dir, color_img)
        sar = cv2.imread(sar_path, cv2.IMREAD_GRAYSCALE)
        color = cv2.imread(color_path, cv2.IMREAD_COLOR)
        sar = cv2.resize(sar, (256, 256))
        color = cv2.resize(color, (256, 256))
        sar = sar / 255.0
        color = color / 255.0
        np.save(os.path.join(output_dir, f"{terrain}_{os.path.splitext(sar_img)[0]}_sar.npy"), sar)
        np.save(os.path.join(output_dir, f"{terrain}_{os.path.splitext(color_img)[0]}_color.npy"), color)
