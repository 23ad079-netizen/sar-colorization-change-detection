import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology, filters
import os

img1_path = r'C:\Users\Student\color\data\processed\train\urban_ROIs1970_fall_s2_143_p1306_color.npy'
img2_path = r'C:\Users\Student\color\data\processed\train\urban_ROIs1970_fall_s2_143_p1315_color.npy'

color1 = np.load(img1_path)   # (H,W,3) in [0,1]
color2 = np.load(img2_path)

# 1) blur slightly to ignore tiny texture changes
from scipy.ndimage import gaussian_filter
blur1 = gaussian_filter(color1, sigma=1.0)
blur2 = gaussian_filter(color2, sigma=1.0)

# 2) difference magnitude
diff = np.abs(blur2 - blur1)
diff_mag = diff.mean(axis=-1)

# 3) automatic threshold (Otsu) + make it stricter
otsu = filters.threshold_otsu(diff_mag)
mask = diff_mag > (otsu * 1.5)      # 1.5x -> only stronger changes

# 4) remove very small and very thin regions
mask = morphology.remove_small_objects(mask, min_size=1000)
mask = morphology.remove_small_holes(mask, area_threshold=1000)
mask = morphology.binary_opening(mask, morphology.disk(3))

# 5) boundaries on cleaned mask
contours = measure.find_contours(mask.astype(float), 0.5)

overlay = color2.copy()
for c in contours:
    c = c.astype(int)
    overlay[c[:, 0], c[:, 1]] = [1.0, 0.0, 0.0]

# 6) show ONLY final change image
# plt.figure(figsize=(4,4))
# plt.title("Detected Major Changes")
# plt.imshow(overlay)
# plt.axis('off')
# plt.tight_layout()
# plt.show()



# 6) show both inputs + final change image
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Color Image 1")
plt.imshow(color1)
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Color Image 2")
plt.imshow(color2)
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Detected Major Changes")
plt.imshow(overlay)
plt.axis('off')

plt.tight_layout()

# save to results/changes folder
out_path = r'C:\Users\Student\color\results\changes\urban_change_example.png'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path)

plt.show()
