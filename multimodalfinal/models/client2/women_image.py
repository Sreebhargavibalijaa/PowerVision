import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from sklearn.neighbors import KernelDensity

# Constants
PATCH_SIZE = 32
device = torch.device("cpu")

# Load and preprocess image
def load_image(path):
    img = Image.open(path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform(img).squeeze(0).numpy()

# Patch decomposition
def decompose_into_patches(image):
    patches = []
    h, w = image.shape
    for i in range(0, h, PATCH_SIZE):
        for j in range(0, w, PATCH_SIZE):
            patch = image[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            if patch.shape == (PATCH_SIZE, PATCH_SIZE):
                patches.append(patch)
    return patches

# Multi-view transform
def apply_multi_view(patch):
    views = []
    patch_uint8 = (patch * 255).astype(np.uint8)
    views.append(cv2.Canny(patch_uint8, 50, 150) / 255.0)
    grad_x = cv2.Sobel(patch_uint8, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(patch_uint8, cv2.CV_64F, 0, 1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    views.append(grad_mag / (grad_mag.max() + 1e-8))
    views.append(cv2.GaussianBlur(patch_uint8, (5, 5), 0) / 255.0)
    lap = cv2.Laplacian(patch_uint8, cv2.CV_64F)
    views.append(lap / (np.max(np.abs(lap)) + 1e-8))
    return np.stack(views, axis=0)

# Fusion model
class FusionModel(torch.nn.Module):
    def __init__(self, num_views=4, patch_size=32):
        super().__init__()
        self.attn = torch.nn.Parameter(torch.ones(num_views))
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(patch_size * patch_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128)
        )

    def forward(self, x):
        w = torch.softmax(self.attn, dim=0)
        x = torch.sum(w.view(-1,1,1)*x, dim=1)
        return self.fc(x)

# KDE scoring
def compute_kde_score(vectors, bandwidth=0.5):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(vectors)
    log_density = kde.score_samples(vectors)
    return log_density

# Mosaic
def apply_mosaic(img, mosaic_size=10):
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // mosaic_size, h // mosaic_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

# Load image
image_path = "/Users/sreebhargavibalija/Desktop/uflallfiles/multimodalfinal/images/womenformal.jpg"
img_orig_color = cv2.imread(image_path)
img_orig_color = cv2.resize(img_orig_color, (256, 256))
img_gray = load_image(image_path)

# Blur
img_blur = cv2.GaussianBlur(img_orig_color, (21, 21), 0)

# Mosaic
img_mosaic = apply_mosaic(img_orig_color, mosaic_size=10)

# PowerVision grayscale output
patches = decompose_into_patches(img_gray)
multi_views = np.array([apply_multi_view(p) for p in patches])
inputs = torch.tensor(multi_views, dtype=torch.float32).to(device)

model = FusionModel().to(device)
model.eval()
with torch.no_grad():
    outputs = model(inputs).cpu().numpy()

conf = compute_kde_score(outputs)
conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)

h, w = img_gray.shape
reconstructed = np.zeros((h, w))
conf_idx = 0
for i in range(0, h, PATCH_SIZE):
    for j in range(0, w, PATCH_SIZE):
        if conf_idx >= len(patches):
            break
        reconstructed[i:i+PATCH_SIZE, j:j+PATCH_SIZE] = patches[conf_idx] * conf_norm[conf_idx]
        conf_idx += 1

reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min() + 1e-8)
reconstructed_color = cv2.cvtColor((reconstructed * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

# Combine all into one image
combined_image = np.hstack([
    cv2.cvtColor(img_orig_color, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(img_mosaic, cv2.COLOR_BGR2RGB),
    reconstructed_color
])

# Plot final combined image
plt.figure(figsize=(16, 6))
plt.imshow(combined_image)
plt.axis('off')
plt.title("Original | Blur | Mosaic | PowerVision")
plt.tight_layout()
# Save final image in the same folder
plt.show()

output_path = "/Users/sreebhargavibalija/Desktop/uflallfiles/multimodalfinal/images/combined_powervision_output.jpg"
cv2.imwrite(output_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
print(f"✅ Saved combined image to: {output_path}")

