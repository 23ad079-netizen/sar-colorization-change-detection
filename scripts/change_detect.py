import numpy as np
import matplotlib.pyplot as plt
from utils import change_overlay

# paths to two SAR .npy files (same area, different times)
img1_path = r'C:\Users\Student\color\data\processed\train\urban_ROIs1970_fall_s1_144_p413_sar.npy'
img2_path = r'C:\Users\Student\color\data\processed\train\urban_ROIs1970_fall_s1_144_p414_sar.npy'

sar1 = np.load(img1_path)
sar2 = np.load(img2_path)

overlay = change_overlay(sar1, sar2, thresh=0.2)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title('Image 1')
plt.imshow(sar1, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title('Image 2')
plt.imshow(sar2, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title('Detected Changes')
plt.imshow(overlay)
plt.axis('off')

plt.tight_layout()
plt.show()
