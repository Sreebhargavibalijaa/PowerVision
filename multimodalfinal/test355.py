import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ====== Step 1: Load Embeddings ======
df_text = pd.read_csv("text_final_all_patients_embeddings.csv", index_col=0)
df_tabular = pd.read_csv("combined_tabular_patient_embeddings.csv", index_col=0)
df_image = pd.read_csv("svlp_dp_patient_embeddings.csv", index_col=0)

# ====== Step 2: Intersect patient_ids ======
common_ids = set(df_text.index) & set(df_tabular.index) & set(df_image.index)
df_text = df_text.loc[common_ids].sort_index()
df_tabular = df_tabular.loc[common_ids].sort_index()
df_image = df_image.loc[common_ids].sort_index()

# ====== Step 3: Rule-Based Labels from patient_id ======
def label_from_id(pid):
    pid_lower = pid.lower()
    if "person" in pid_lower:
        return 1  # pneumonia
    elif "normal" in pid_lower or "img" in pid_lower:
        return 0  # normal
    else:
        return None  # unknown

labels = [label_from_id(pid) for pid in df_text.index]
labels = np.array(labels)

# Filter out any unknown labels (None)
valid_indices = [i for i, y in enumerate(labels) if y is not None]
df_text = df_text.iloc[valid_indices]
df_tabular = df_tabular.iloc[valid_indices]
df_image = df_image.iloc[valid_indices]
labels = labels[valid_indices]

# ====== Step 4: Fuse Embeddings (Quantum Superposition) ======
w1, w2, w3 = 1/3, 1/3, 1/3
fused_embeddings = w1 * df_text.values + w2 * df_tabular.values + w3 * df_image.values

# ====== Step 5: Train/Test Split ======
X_train, X_test, y_train, y_test = train_test_split(fused_embeddings, labels, test_size=0.2, random_state=42)

# ====== Step 6: Train Classifier ======
clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)

# ====== Step 7: Evaluate ======
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")
