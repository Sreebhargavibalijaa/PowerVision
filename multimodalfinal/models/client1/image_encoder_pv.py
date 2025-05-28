# 📦 Imports
#Lipschitz-based Power Learning with Patchwise Decomposition + Local KDE Calibration
#(aka PowerVision, an extension of PowerLearn for images)
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from torch.utils.data import DataLoader, Dataset
from opacus import PrivacyEngine
import random
import cv2
import matplotlib.pyplot as plt
MAX_IMAGES = 10
image_counter = 0

# ✅ 1. Parameters
PATCH_SIZE = 32
NOISE_MULTIPLIER = 1.0
MAX_GRAD_NORM = 1.0
EPSILON_TARGET = 2.0
DELTA_TARGET = 1e-5
BATCH_SIZE = 8
EPOCHS = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 2. Paths
CLIENT_PATH = "/Users/sreebhargavibalija/Desktop/uflallfiles/UFM3- FederatedAgents/chestx-ray/clients/reduced_client1"

# ✅ 3. Patch Pool
ALL_REFERENCE_PATCHES = []

def load_image(path):
    img = Image.open(path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform(img).squeeze(0).numpy()

def decompose_into_patches(image):
    patches = []
    h, w = image.shape
    for i in range(0, h, PATCH_SIZE):
        for j in range(0, w, PATCH_SIZE):
            patch = image[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            if patch.shape == (PATCH_SIZE, PATCH_SIZE):
                patches.append(patch)
    return patches

def prepare_reference_patches(client_path):
    global ALL_REFERENCE_PATCHES
    for patient_folder in os.listdir(client_path):
        patient_path = os.path.join(client_path, patient_folder)
        if not os.path.isdir(patient_path):
            continue
        for root, _, files in os.walk(patient_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image = load_image(os.path.join(root, file))
                    patches = decompose_into_patches(image)
                    ALL_REFERENCE_PATCHES.extend(patches)

def random_patch_noise(patch):
    if not ALL_REFERENCE_PATCHES:
        raise ValueError("No reference patches loaded. Run prepare_reference_patches() first.")
    return random.choice(ALL_REFERENCE_PATCHES)
from sklearn.neighbors import KernelDensity

def compute_kde_score(vectors, bandwidth=0.5):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(vectors)
    log_density = kde.score_samples(vectors)
    return log_density  # Higher => more confident (less uncertain)

def apply_multi_view(patch):
    views = []
    patch_uint8 = (patch * 255).astype(np.uint8)
    # 1. Canny Edge
    views.append(cv2.Canny(patch_uint8, 50, 150) / 255.0)
    # 2. Gradie Magnitude
    grad_x = cv2.Sobel(patch_uint8, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(patch_uint8, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag_max = grad_mag.max()
    views.append(grad_mag / grad_mag_max if grad_mag_max > 0 else grad_mag)
    # Gaussian Blur
    blur = cv2.GaussianBlur(patch_uint8, (5, 5), 0)
    views.append(blur / 255.0)


    # 4. Laplacian
    laplacian = cv2.Laplacian(patch_uint8, cv2.CV_64F)
    laplacian_max = np.max(np.abs(laplacian))
    views.append(laplacian / laplacian_max if laplacian_max > 0 else laplacian)

    return np.stack(views, axis=0)  # Shape: (4, PATCH_SIZE, PATCH_SIZE)

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

def lipschitz_regularization(model, x):
    x.requires_grad_(True)
    out = model(x)
    grad = torch.autograd.grad(outputs=out, inputs=x,
                               grad_outputs=torch.ones_like(out),
                               create_graph=True)[0]
    return grad.norm(2)
def plot_patch_confidence_heatmap(image_np, patch_confidences, patch_size=32, title="Patch Confidence Heatmap", save_path=None):
    h, w = image_np.shape
    patch_grid_h = h // patch_size
    patch_grid_w = w // patch_size
    heatmap = np.zeros((patch_grid_h, patch_grid_w))
    idx = 0
    for i in range(patch_grid_h):
        for j in range(patch_grid_w):
            if idx < len(patch_confidences):
                heatmap[i, j] = patch_confidences[idx]
                idx += 1
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    plt.figure(figsize=(6, 6))
    plt.imshow(image_np, cmap='gray')
    plt.imshow(heatmap_resized, cmap='hot', alpha=0.5)
    plt.title(title)
    plt.axis('off')


    
    plt.colorbar(label='Confidence (KDE-based)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


class PatchDataset(Dataset):
    def __init__(self, patches, labels):
        self.patches = patches
        self.labels = labels
    def __len__(self): return len(self.patches)
    def __getitem__(self, idx):
        return torch.tensor(self.patches[idx], dtype=torch.float32), self.labels[idx]

prepare_reference_patches(CLIENT_PATH)

patient_embeddings, patient_labels, patient_ids = [], [], []
losses_per_epoch, epsilons_per_epoch = [], []

for patient_folder in tqdm(os.listdir(CLIENT_PATH), desc="Processing Patients"):
    patient_path = os.path.join(CLIENT_PATH, patient_folder)
    if not os.path.isdir(patient_path): continue

    for root, _, files in os.walk(patient_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image = load_image(os.path.join(root, file))
                patches = decompose_into_patches(image)
                mixed_patches = [random_patch_noise(p) if random.random() < 0.2 else p for p in patches]
                X = np.array([apply_multi_view(p) for p in mixed_patches])
                y = 1 if "person" in file.lower() else 0
                dataset = PatchDataset(X, [y]*len(X))
                loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

                model = FusionModel().to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                privacy_engine = PrivacyEngine()
                model, optimizer, loader = privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=loader,
                    noise_multiplier=NOISE_MULTIPLIER,
                    max_grad_norm=MAX_GRAD_NORM,
                )

                for epoch in range(EPOCHS):
                    running_loss = 0
                    for inputs, labels_batch in loader:
                        inputs, labels_batch = inputs.to(device), labels_batch.to(device)
                        optimizer.zero_grad(set_to_none=True)
                        outputs = model(inputs)
                        ce_loss = nn.CrossEntropyLoss()(outputs, labels_batch)
                        lp_loss = lipschitz_regularization(model, inputs)
                        loss = ce_loss + 0.1 * lp_loss  # lambda=0.1
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                    avg_loss = running_loss / len(loader)
                    epsilon = privacy_engine.get_epsilon(delta=DELTA_TARGET)
                    losses_per_epoch.append(avg_loss)
                    epsilons_per_epoch.append(epsilon)

                model.eval()
                all_patch_outputs, all_patch_inputs = [], []

                with torch.no_grad():
                    for inputs, _ in loader:
                        outputs = model(inputs.to(device)).cpu()
                        all_patch_outputs.append(outputs)
                        all_patch_inputs.append(inputs.cpu())

                patch_embeddings = torch.cat(all_patch_outputs, dim=0).numpy()
                patch_inputs = torch.cat(all_patch_inputs, dim=0).numpy()

                # 🧠 Apply KDE calibration
                kde_scores = compute_kde_score(patch_embeddings)
                confidences = (kde_scores - np.min(kde_scores)) / (np.max(kde_scores) - np.min(kde_scores) + 1e-8)  # Normalize to [0,1]

                # ✅ Apply confidence-weighted aggregation
                weighted_embedding = np.average(patch_embeddings, axis=0, weights=confidences)

                # ✅ Plot confidence heatmap for this image
                original_image = image  # Already loaded as grayscale numpy array
                # Save heatmap after computing confidences
                heatmap_dir = "c1_dp_prompt_patient_embeddings_csv/heatmaps"
                os.makedirs(heatmap_dir, exist_ok=True)
                heatmap_path = os.path.join(heatmap_dir, f"{os.path.splitext(file)[0]}_heatmap.png")
                plot_patch_confidence_heatmap(original_image, confidences, save_path=heatmap_path)
                # Limit number of saved heatmaps
                image_counter += 1
                # ✅ Save aggregated results
                patient_embeddings.append(weighted_embedding)
                patient_labels.append(y)
                patient_ids.append(os.path.splitext(file)[0])



# Save embeddings
os.makedirs("c1_dp_prompt_patient_embeddings_csv", exist_ok=True)
columns = ['patient_id'] + [f"dim_{i}" for i in range(len(patient_embeddings[0]))] + ['label']
final_df = pd.DataFrame([[pid]+list(emb)+[label] for pid, emb, label in zip(patient_ids, patient_embeddings, patient_labels)], columns=columns)
final_df.to_csv("c1_image_final_all_patients_embeddings.csv", index=False)
print("🎯 Success! Saved embeddings.")

# Plot
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(range(1, len(losses_per_epoch)+1), losses_per_epoch, marker='o')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.subplot(1,2,2)
plt.plot(range(1, len(epsilons_per_epoch)+1), epsilons_per_epoch, marker='o', color='green')
plt.title('Privacy ε vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Epsilon (ε)')
plt.grid()
plt.tight_layout()
plt.show()
