import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, log_loss
)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

print("🚀 Starting Multi-Modal Federated Patient Classification Pipeline")

# ====== Config ======
sigma_text = 0.08     # DP2-style noise
sigma_tabular = 0.1   # PowerTabular-style noise
sigma_image = 0.12    # PowerVision-style noise
random_seed = 42
test_split_ratio = 0.2

np.random.seed(random_seed)

# ====== Step 1: Load and Preprocess ======
def load_and_average_embeddings(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.lower()
    return df.groupby(df.index).mean()

print("📥 Loading embeddings...")
df_text = load_and_average_embeddings("/Users/sreebhargavibalija/Desktop/uflallfiles/multimodalfinal/models/client1/c1_text_final_all_patients_embeddings.csv")
df_tabular = load_and_average_embeddings("/Users/sreebhargavibalija/Desktop/uflallfiles/multimodalfinal/models/client1/c1_combined_tabular_patient_embeddings.csv")
df_image = load_and_average_embeddings("/Users/sreebhargavibalija/Desktop/uflallfiles/multimodalfinal/models/client1/c1_image_final_all_patients_embeddings.csv")

print(f"✅ Unique Text IDs: {len(df_text)}")
print(f"✅ Unique Tabular IDs: {len(df_tabular)}")
print(f"✅ Unique Image IDs: {len(df_image)}")

# ====== Step 2: Filter by Common Patients ======
common_ids = sorted(set(df_text.index) & set(df_tabular.index) & set(df_image.index))
print(f"✅ Common patient IDs: {len(common_ids)}")
if len(common_ids) == 0:
    raise ValueError("❌ No common patient IDs found!")

df_text = df_text.loc[common_ids].sort_index()
df_tabular = df_tabular.loc[common_ids].sort_index()
df_image = df_image.loc[common_ids].sort_index()

# ====== Step 3: Labeling ======
def label_from_id(pid):
    pid_lower = pid.lower()
    if "person" in pid_lower:
        return 1
    elif "normal" in pid_lower or "img" in pid_lower:
        return 0
    else:
        return None

labels = np.array([label_from_id(pid) for pid in common_ids])
valid_indices = [i for i, y in enumerate(labels) if y is not None]

df_text = df_text.iloc[valid_indices]
df_tabular = df_tabular.iloc[valid_indices]
df_image = df_image.iloc[valid_indices]
labels = labels[valid_indices].astype(int)
valid_ids = [common_ids[i] for i in valid_indices]

# ====== Step 4: Apply Noise ======
def apply_gaussian_noise(data, sigma):
    return data + np.random.normal(0.0, sigma, data.shape)

dp_text = apply_gaussian_noise(df_text.values, sigma=sigma_text)
dp_tabular = apply_gaussian_noise(df_tabular.values, sigma=sigma_tabular)
dp_image = apply_gaussian_noise(df_image.values, sigma=sigma_image)

# ====== Step 5: Fuse ======
fused_embeddings = np.concatenate([dp_text, dp_tabular, dp_image], axis=1)
print(f"📐 Fused Embedding Shape: {fused_embeddings.shape}")

# ====== Step 6: Train/Test Split ======
X_train, X_test, y_train, y_test = train_test_split(
    fused_embeddings, labels, test_size=test_split_ratio, random_state=random_seed, stratify=labels
)

# ====== Step 7: Train Logistic Regression ======
clf = LogisticRegression(penalty='l2', C=0.01, solver='lbfgs', max_iter=500, random_state=random_seed)
clf.fit(X_train, y_train)

# ====== Step 8: Evaluation ======
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
logloss = log_loss(y_test, y_proba)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n🎯 Evaluation Metrics")
print(f"✅ Accuracy: {acc * 100:.2f}%")
print(f"✅ Precision: {prec * 100:.2f}%")
print(f"✅ Recall: {rec * 100:.2f}%")
print(f"✅ F1 Score: {f1 * 100:.2f}%")
print(f"✅ Log Loss: {logloss:.4f}")
print(f"✅ Confusion Matrix:\n{conf_matrix}")

# ====== Step 9: Save Files ======
base_path = "/Users/sreebhargavibalija/Desktop/uflallfiles/multimodalfinal/models/client1/"
os.makedirs(base_path, exist_ok=True)

np.save(base_path + "c1_final_fused_embeddings.npy", fused_embeddings)
np.save(base_path + "c1_final_labels.npy", labels)

metrics_df = pd.DataFrame({
    "Accuracy (%)": [acc * 100],
    "Precision (%)": [prec * 100],
    "Recall (%)": [rec * 100],
    "F1 Score (%)": [f1 * 100],
    "Log Loss": [logloss]
})
metrics_df.to_csv(base_path + "c1_metrics.csv", index=False)

conf_df = pd.DataFrame(conf_matrix, columns=["Predicted Negative", "Predicted Positive"],
                       index=["Actual Negative", "Actual Positive"])
conf_df.to_csv(base_path + "c1_confusion_matrix.csv")

fused_df = pd.DataFrame(fused_embeddings)
fused_df['label'] = labels
fused_df.to_csv(base_path + "c1_embeddings_with_labels.csv", index=False)

print("✅ All outputs saved successfully!")

# ====== Step 10 (Optional): t-SNE Visualization ======
print("📊 Generating t-SNE visualization...")
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=random_seed)
tsne_proj = tsne.fit_transform(fused_embeddings)

plt.figure(figsize=(10, 6))
plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=labels, cmap="coolwarm", alpha=0.7)
plt.title("t-SNE Projection of Fused Embeddings")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.savefig(base_path + "c1_tsne_visualization.png")
plt.show()

print("🎉 Pipeline complete!")
