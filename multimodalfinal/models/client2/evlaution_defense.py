import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# 📂 Load patient embeddings
df = pd.read_csv("/Users/sreebhargavibalija/Desktop/uflallfiles/multimodalfinal/models/client2/c2_image_final_all_patients_embeddings.csv")

# ========== 1. Membership Inference Attack ==========
shadow_df, _ = train_test_split(df, test_size=0.5, stratify=df['label'], random_state=42)
shadow_df['membership'] = 1
non_member_shadow = shadow_df.sample(frac=0.5, random_state=42).copy()
non_member_shadow['membership'] = 0
mif_data = pd.concat([shadow_df, non_member_shadow]).sample(frac=1.0, random_state=42)

X_mif = mif_data.drop(columns=['patient_id', 'label', 'membership'])
y_mif = mif_data['membership']
X_train, X_test, y_train, y_test = train_test_split(X_mif, y_mif, test_size=0.3, random_state=42)

mia_clf = LogisticRegression(max_iter=1000)
mia_clf.fit(X_train, y_train)
mia_preds = mia_clf.predict(X_test)
mia_scores = mia_clf.predict_proba(X_test)[:, 1]

mia_acc = accuracy_score(y_test, mia_preds)
mia_auc = roc_auc_score(y_test, mia_scores)

# ========== 2. Feature Leakage ==========
X_feat = df.drop(columns=['patient_id', 'label'])
y_feat = df['label']
Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_feat, y_feat, test_size=0.3, random_state=42)

clf_feat = LogisticRegression(max_iter=1000)
clf_feat.fit(Xf_train, yf_train)
feat_preds = clf_feat.predict(Xf_test)
feat_acc = accuracy_score(yf_test, feat_preds)
feat_auc = roc_auc_score(yf_test, clf_feat.predict_proba(Xf_test)[:, 1])

# t-SNE Visualization
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X_feat)
df_tsne = pd.DataFrame(X_embedded, columns=["x", "y"])
df_tsne["label"] = y_feat.values

plt.figure(figsize=(6, 6))
sns.scatterplot(data=df_tsne, x="x", y="y", hue="label", palette="coolwarm", alpha=0.7)
plt.title("t-SNE Visualization: Feature Leakage")
plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_leakage_plot.png")

# ========== 3. Gradient Norm Proxy ==========
gradient_norms = [np.linalg.norm(x) for x in X_feat.values]
mean_grad_norm = np.mean(gradient_norms)

# ========== 4. Utility Metrics ==========
utility_acc = accuracy_score(yf_test, feat_preds)
utility_prec = precision_score(yf_test, feat_preds)
utility_rec = recall_score(yf_test, feat_preds)
utility_f1 = f1_score(yf_test, feat_preds)
utility_auc = feat_auc  # already computed above

# ========== 5. Report ==========
print("\n🔐 PowerVision Defense & Utility Evaluation")
print("📌 Membership Inference:")
print(f"    • Accuracy:     {mia_acc:.4f}")
print(f"    • AUC:          {mia_auc:.4f}")
print("🧠 Feature Leakage:")
print(f"    • Accuracy:     {feat_acc:.4f}")
print(f"    • AUC:          {feat_auc:.4f}")
print("🧪 Gradient Inversion Risk:")
print(f"    • Mean Grad Norm: {mean_grad_norm:.4f}")
print("📈 Utility (Classification):")
print(f"    • Accuracy:     {utility_acc:.4f}")
print(f"    • Precision:    {utility_prec:.4f}")
print(f"    • Recall:       {utility_rec:.4f}")
print(f"    • F1 Score:     {utility_f1:.4f}")
print(f"    • ROC AUC:      {utility_auc:.4f}")
print("\n📊 t-SNE plot saved as: tsne_leakage_plot.png")
