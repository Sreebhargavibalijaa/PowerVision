# 📦 Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
# ✅ 1. Load the generated CSV
df = pd.read_csv("/Users/sreebhargavibalija/Desktop/uflallfiles/multimodalfinal/models/client1/c1_text_final_all_patients_embeddings.csv")

# ✅ 2. Prepare X, y
X = df[[col for col in df.columns if col.startswith('dim_')]].values
y = df['label'].values

# ✅ 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ 4. Torch Dataset and Loader
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# ✅ 5. Define a Deep MLP Model
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # 2 classes: pneumonia or normal

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLPClassifier(X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ 6. Training Loop
for epoch in range(20):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# ✅ 7. Evaluation
model.eval()
with torch.no_grad():
    y_preds = []
    y_true = []
    for inputs, labels in test_loader:
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        y_preds.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# ✅ 8. Results
print("✅ Accuracy:", accuracy_score(y_true, y_preds))
print("\n✅ Classification Report:\n", classification_report(y_true, y_preds))
# 📦 Imports
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ✅ 1. Load CSV
df = pd.read_csv("/Users/sreebhargavibalija/Desktop/uflallfiles/multimodalfinal/models/client1/c1_text_final_all_patients_embeddings.csv")

# ✅ 2. Prepare X, y
X = df[[col for col in df.columns if col.startswith('dim_')]].values
y = df['label'].values

# ✅ 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ 4. Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# ✅ 5. Predict and Evaluate
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))
