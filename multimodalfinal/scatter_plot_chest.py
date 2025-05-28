import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def load_and_process(client_folder_path, prefix, n_components=128):
    X = np.load(f"{client_folder_path}/{prefix}_final_fused_embeddings.npy")
    y = np.load(f"{client_folder_path}/{prefix}_final_labels.npy")

    # Add Gaussian noise
    X_noisy = X + np.random.normal(0, 0.05, size=X.shape)

    # Apply PCA
    X_reduced = PCA(n_components=n_components).fit_transform(X_noisy)

    # Stratified 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, train_size=0.8, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test

# Load and preprocess data from all three clients
c1_train_X, c1_test_X, c1_train_y, c1_test_y = load_and_process(
    "/Users/sreebhargavibalija/Desktop/uflallfiles/multimodalfinal/models/client1", "c1")
c2_train_X, c2_test_X, c2_train_y, c2_test_y = load_and_process(
    "/Users/sreebhargavibalija/Desktop/uflallfiles/multimodalfinal/models/client2", "c2")
c3_train_X, c3_test_X, c3_train_y, c3_test_y = load_and_process(
    "/Users/sreebhargavibalija/Desktop/uflallfiles/multimodalfinal/models/client3", "c3")

# Combine all clients' data
X_train_global = np.vstack([c1_train_X, c2_train_X, c3_train_X])
y_train_global = np.concatenate([c1_train_y, c2_train_y, c3_train_y])

X_test_global = np.vstack([c1_test_X, c2_test_X, c3_test_X])
y_test_global = np.concatenate([c1_test_y, c2_test_y, c3_test_y])

print(f"✅ Global Train Shape: {X_train_global.shape}")
print(f"✅ Global Test Shape: {X_test_global.shape}")

# Train model
clf = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000, C=0.5)
clf.fit(X_train_global, y_train_global)

# Predict
y_pred = clf.predict(X_test_global)
y_prob = clf.predict_proba(X_test_global)[:, 1]

# Other metrics
acc = accuracy_score(y_test_global, y_pred)
prec = precision_score(y_test_global, y_pred)
rec = recall_score(y_test_global, y_pred)
f1 = f1_score(y_test_global, y_pred)

print("\n📊 Federated-Like Evaluation (Client1 + Client2 + Client3):")
print(f"✅ Accuracy: {acc*100:.2f}%")
print(f"✅ Precision: {prec*100:.2f}%")
print(f"✅ Recall: {rec*100:.2f}%")
print(f"✅ F1 Score: {f1*100:.2f}%")

# Plot 2D PCA scatter plot with color gradient for labels
# First reduce test data to 2D for visualization
pca_2d = PCA(n_components=2)
X_test_pca_2d = pca_2d.fit_transform(X_test_global)

plt.figure(figsize=(8, 6))
norm = Normalize(vmin=0, vmax=1)
cmap = cm.coolwarm
scatter = plt.scatter(
    X_test_pca_2d[:, 0], X_test_pca_2d[:, 1],
    c=y_test_global,
    cmap=cmap,
    norm=norm,
    alpha=0.7,
    edgecolors='none',
    linewidths=0.5
)
cbar = plt.colorbar(scatter)
cbar.set_label("Label", fontsize=12)
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.title("PCA of Fused DP Embeddings (Image + Text + Tabular)", fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.show()
