# 📦 Imports
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
from torch.utils.data import DataLoader, Dataset
from opacus import PrivacyEngine
import random
import cv2

# ✅ 1. Parameters
PATCH_SIZE = 32
NOISE_MULTIPLIER = 1.0  # DP noise strength
MAX_GRAD_NORM = 1.0
EPSILON_TARGET = 2.0
DELTA_TARGET = 1e-5
BATCH_SIZE = 32
EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 2. Paths
CLIENT_PATH = "/Users/sreebhargavibalija/Desktop/uflallfiles/UFM3- FederatedAgents/chestx-ray/clients/client1"

# ✅ 3. Global Patch Pool
ALL_REFERENCE_PATCHES = []

def load_image(path):
    img = Image.open(path).convert('L')  # Grayscale
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
    """
    Collect patches from all patients for realistic patch replacement.
    """
    global ALL_REFERENCE_PATCHES
    for patient_folder in os.listdir(client_path):
        patient_path = os.path.join(client_path, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        for root, _, files in os.walk(patient_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_path = os.path.join(root, file)
                    image = load_image(image_path)
                    patches = decompose_into_patches(image)
                    ALL_REFERENCE_PATCHES.extend(patches)

    print(f"✅ Collected {len(ALL_REFERENCE_PATCHES)} reference patches for mixing.")

def random_patch_noise(patch):
    """
    Instead of random noise, replace with a real random patch from other images.
    """
    if not ALL_REFERENCE_PATCHES:
        raise ValueError("❌ No reference patches loaded. Run prepare_reference_patches() first.")
    ref_patch = random.choice(ALL_REFERENCE_PATCHES)
    return ref_patch

def apply_multi_view(patch):
    """
    Create multiple views of a patch.
    """
    views = []
    patch_uint8 = (patch * 255).astype(np.uint8)

    # Canny Edge
    edges = cv2.Canny(patch_uint8, 50, 150)
    views.append(edges / 255.0)

    # Sobel Gradients
    grad_x = cv2.Sobel(patch_uint8, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(patch_uint8, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    views.append(grad_magnitude / grad_magnitude.max())

    # Gaussian Blur (low frequency)
    blur = cv2.GaussianBlur(patch_uint8, (5, 5), 0)
    views.append(blur / 255.0)

    # Laplacian Filter
    laplacian = cv2.Laplacian(patch_uint8, cv2.CV_64F)
    views.append(laplacian / np.max(np.abs(laplacian)))

    return np.stack(views, axis=0)  # (num_views, PATCH_SIZE, PATCH_SIZE)

# ✅ 4. Fusion Model
class FusionModel(nn.Module):
    def __init__(self, num_views=4, patch_size=32):
        super(FusionModel, self).__init__()
        self.attention_weights = nn.Parameter(torch.ones(num_views))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(patch_size * patch_size, 512),  # ✅ was 4096, now 1024
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        # x shape: (batch_size, num_views, patch_size, patch_size)
        weights = torch.softmax(self.attention_weights, dim=0)
        weighted_sum = torch.sum(weights.view(-1,1,1) * x, dim=1)  # Shape: (batch_size, patch_size, patch_size)
        out = self.fc(weighted_sum)
        return out


# ✅ 5. Dataset Class
class PatchDataset(Dataset):
    def __init__(self, patches, labels):
        self.patches = patches
        self.labels = labels

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return torch.tensor(self.patches[idx], dtype=torch.float32), self.labels[idx]

# ✅ 6. Main Processing
# Collect reference patches first
prepare_reference_patches(CLIENT_PATH)

patient_embeddings = []
patient_labels = []
patient_ids = []

for patient_folder in tqdm(os.listdir(CLIENT_PATH), desc="Processing Patients"):
    patient_path = os.path.join(CLIENT_PATH, patient_folder)
    if not os.path.isdir(patient_path):
        continue

    for root, _, files in os.walk(patient_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(root, file)
                image = load_image(image_path)

                patches = decompose_into_patches(image)

                # Randomly mix some patches
                mixed_patches = []
                for patch in patches:
                    if np.random.rand() < 0.2:
                        mixed_patches.append(random_patch_noise(patch))
                    else:
                        mixed_patches.append(patch)

                # Apply multi-view
                multi_view_patches = [apply_multi_view(patch) for patch in mixed_patches]

                X = np.array(multi_view_patches)
                y = 1 if "person" in file.lower() else 0

                # Train a fusion model
                dataset = PatchDataset(X, [y]*len(X))
                loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

                fusion_model = FusionModel().to(device)
                optimizer = optim.Adam(fusion_model.parameters(), lr=0.001)

                # ✅ Attach DP Engine
                privacy_engine = PrivacyEngine()
                fusion_model, optimizer, loader = privacy_engine.make_private(
                    module=fusion_model,
                    optimizer=optimizer,
                    data_loader=loader,
                    noise_multiplier=NOISE_MULTIPLIER,
                    max_grad_norm=MAX_GRAD_NORM,
                )

                # Train fusion model
                fusion_model.train()
                for epoch in range(EPOCHS):
                    for inputs, labels_batch in loader:
                        inputs = inputs.to(device)
                        labels_batch = labels_batch.to(device)

                        optimizer.zero_grad()
                        outputs = fusion_model(inputs)
                        loss = nn.CrossEntropyLoss()(outputs, labels_batch)
                        loss.backward()
                        optimizer.step()

                # Generate patient embedding
                fusion_model.eval()
                with torch.no_grad():
                    all_outputs = []
                    for inputs, _ in loader:
                        inputs = inputs.to(device)
                        outputs = fusion_model(inputs)
                        all_outputs.append(outputs.cpu())

                    patient_embedding = torch.mean(torch.cat(all_outputs, dim=0), dim=0)

                # Store results
                patient_embeddings.append(patient_embedding.numpy())
                patient_labels.append(y)
                patient_ids.append(os.path.splitext(file)[0])

                # Report Privacy
                epsilon = privacy_engine.get_epsilon(delta=DELTA_TARGET)
                # print(f"✅ Final ε for {file}: {epsilon:.2f}")

# ✅ 7. Save final patient embeddings
os.makedirs("c1_dp_prompt_patient_embeddings_csv", exist_ok=True)
embedding_dim = len(patient_embeddings[0])
columns = ['patient_id'] + [f"dim_{i}" for i in range(embedding_dim)] + ['label']

final_df = pd.DataFrame(
    [[pid] + list(emb) + [label] for pid, emb, label in zip(patient_ids, patient_embeddings, patient_labels)],
    columns=columns
)

save_path = os.path.join("c1_dp_prompt_patient_embeddings_csv", "c1_image_final_all_patients_embeddings.csv")
final_df.to_csv(save_path, index=False)

print(f"🎯 Success! Saved {len(patient_embeddings)} patient embeddings to {save_path}")
import matplotlib.pyplot as plt

# ✅ Track during training
losses_per_epoch = []
epsilons_per_epoch = []

# (Inside your training loop, after each epoch)

for epoch in range(EPOCHS):
    running_loss = 0.0
    for inputs, labels_batch in loader:
        inputs = inputs.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()
        outputs = fusion_model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    losses_per_epoch.append(avg_loss)

    # Track epsilon
    epsilon = privacy_engine.get_epsilon(delta=DELTA_TARGET)
    epsilons_per_epoch.append(epsilon)

    print(f"🔵 Epoch {epoch+1}: Loss={avg_loss:.4f}, ε={epsilon:.2f}")

# ✅ After training, plot
plt.figure(figsize=(12,5))

# Loss Plot
plt.subplot(1,2,1)
plt.plot(range(1, EPOCHS+1), losses_per_epoch, marker='o')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()

# Epsilon Plot
plt.subplot(1,2,2)
plt.plot(range(1, EPOCHS+1), epsilons_per_epoch, marker='o', color='green')
plt.title('Privacy ε vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Epsilon (ε)')
plt.grid()

plt.tight_layout()
plt.show()
