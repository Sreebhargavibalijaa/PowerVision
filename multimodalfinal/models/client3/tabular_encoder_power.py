# 📦 Imports
import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from opacus import PrivacyEngine

# ✅ 1. Paths
CLIENT_PATHS = [
    "/Users/sreebhargavibalija/Desktop/uflallfiles/UFM3- FederatedAgents/chestx-ray/clients/reduced_client3"]

# ✅ 2. Generate synthetic tabular data across multiple clients
def generate_tabular_data(client_paths):
    data_records = []
    labels = []
    patient_ids = []

    for base_path in client_paths:
        for label_folder, label_value in [("normal", 0), ("pneumonia", 1)]:
            folder_path = os.path.join(base_path, label_folder)
            if not os.path.exists(folder_path):
                continue
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
                    patient_id = filename.split('.')[0]  # ⛔ no client prefix # make patient_id unique across clients

                    # Generate synthetic tabular features
                    record = {
                        "heart_rate": random.randint(60, 100),
                        "oxygen_saturation": random.uniform(90, 100),
                        "temperature": random.uniform(36.0, 39.0),
                        "age": random.randint(1, 90),
                        "blood_pressure": random.randint(80, 140)
                    }

                    data_records.append(record)
                    labels.append(label_value)
                    patient_ids.append(patient_id)

    df = pd.DataFrame(data_records)
    df["label"] = labels
    df["patient_id"] = patient_ids
    return df

# ✅ 3. Power Learning Model
class PowerLearningModel(nn.Module):
    def __init__(self, input_dim, embed_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, embed_dim)  # Embedding layer
        self.out = nn.Linear(embed_dim, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        embedding = torch.relu(self.fc2(x))
        out = self.out(embedding)
        return out, embedding

# ✅ 4. Train Model with Privacy
def train_power_model(df, target_epsilon=5.0, target_delta=1e-5):
    features = ["heart_rate", "oxygen_saturation", "temperature", "age", "blood_pressure"]
    X = df[features].values
    y = df["label"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = PowerLearningModel(input_dim=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
    )

    for epoch in range(10):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            out, _ = model(batch_X)
            loss = criterion(out, batch_y)

            # Apply random power scaling
            scaled_loss = loss * random.uniform(0.8, 1.2)
            scaled_loss.backward()
            optimizer.step()

    epsilon = privacy_engine.get_epsilon(delta=target_delta)
    print(f"✅ Achieved privacy: ε = {epsilon:.2f} at δ = {target_delta}")
    return model, scaler

# ✅ 5. Extract Embeddings
@torch.no_grad()
def generate_embeddings(model, scaler, df):
    features = ["heart_rate", "oxygen_saturation", "temperature", "age", "blood_pressure"]
    X = df[features].values
    X = scaler.transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    embeddings = {}

    model.eval()
    for i, patient_id in enumerate(df["patient_id"]):
        _, embedding = model(X_tensor[i].unsqueeze(0))
        embeddings[patient_id] = embedding.squeeze(0).numpy()
    return embeddings

# ✅ 6. Pipeline
if __name__ == "__main__":
    # Combine all clients
    df = generate_tabular_data(CLIENT_PATHS)

    print(f"✅ Total patient records generated: {len(df)}")

    model, scaler = train_power_model(df)
    patient_embeddings = generate_embeddings(model, scaler, df)

    # Save embeddings
    embeddings_df = pd.DataFrame.from_dict(patient_embeddings, orient="index")
    embeddings_df.to_csv("c3_combined_tabular_patient_embeddings.csv")

    print("✅ Done! Generated embeddings saved to combined_tabular_patient_embeddings.csv.")
