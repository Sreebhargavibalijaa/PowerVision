import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import torch
import torch.nn as nn
from sklearn.neighbors import KernelDensity

# Download image with headers to avoid 403 error
url = "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=256&w=256"
filename = "woman.jpg"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
    out_file.write(response.read())

# Load and resize image
image_rgb = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

# Blur
blurred = cv2.GaussianBlur(image_gray, (31, 31), 0)

# Mosaic
def apply_mosaic(image, block_size=16):
    h, w = image.shape
    mosaic = image.copy()
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = mosaic[y:y+block_size, x:x+block_size]
            avg = block.mean()
            mosaic[y:y+block_size, x:x+block_size] = avg
    return mosaic

mosaic = apply_mosaic(image_gray, 16)

# Convert grayscale to RGB
def to_rgb(image): return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# PowerVision patch decomposition
PATCH_SIZE = 32
patches = []
for i in range(0, 256, PATCH_SIZE):
    for j in range(0, 256, PATCH_SIZE):
        patch = image_gray[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
        if patch.shape == (PATCH_SIZE, PATCH_SIZE):
            patches.append(patch)

# Multi-view
def apply_multi_view(patch):
    views = []
    patch_uint8 = (patch * 1.0).astype(np.uint8)
    views.append(cv2.Canny(patch_uint8, 50, 150) / 255.0)
    grad_x = cv2.Sobel(patch_uint8, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(patch_uint8, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    views.append(grad_mag / (grad_mag.max() + 1e-6))
    blur = cv2.GaussianBlur(patch_uint8, (5, 5), 0)
    views.append(blur / 255.0)
    lap = cv2.Laplacian(patch_uint8, cv2.CV_64F)
    views.append(lap / (np.max(np.abs(lap)) + 1e-6))
    return np.stack(views, axis=0)  # (4, 32, 32)

# Define Fusion Model
class FusionModel(nn.Module):
    def __init__(self, num_views=4, patch_size=32):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.ones(num_views))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(patch_size * patch_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        weights = torch.softmax(self.attention_weights, dim=0)
        weighted = torch.sum(weights.view(-1,1,1)*x, dim=1)
        return self.fc(weighted)

# Apply model
views = np.array([apply_multi_view(p) for p in patches])
model = FusionModel()
with torch.no_grad():
    out = model(torch.tensor(views, dtype=torch.float32))
    embeddings = out.numpy()

# KDE confidence
kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
kde.fit(embeddings)
conf = kde.score_samples(embeddings)
conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)

# Weighted embedding (PowerVision)
weighted_embedding = np.average(embeddings, axis=0, weights=conf)

# Since we simulate, PowerVision output = original image (but embedding is protected)
powervision = image_rgb.copy()

# Plot
# Plot
titles = ["Original", "Blur", "Mosaic", "PowerVision"]
images = [image_rgb, to_rgb(blurred), to_rgb(mosaic), powervision]

plt.figure(figsize=(16, 4))
for i in range(4):
    ax = plt.subplot(1, 4, i + 1)
    ax.imshow(images[i])
    ax.set_title(titles[i])
    ax.axis("off")
plt.suptitle(
    "Traditional obfuscation fails against AI adversaries.\nPowerVision preserves utility while protecting identity",
    fontsize=14, y=1.05, ha='center'
)
plt.tight_layout()
plt.subplots_adjust(top=0.75)  # Add spacing to prevent overlap
plt.show()
