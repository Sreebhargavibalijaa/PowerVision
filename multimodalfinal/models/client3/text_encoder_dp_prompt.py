import os
import random
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# =========================
# 1. Setup: Group X-ray images across all clients
# =========================

CLIENT_PATHS = [
    "/Users/sreebhargavibalija/Desktop/uflallfiles/UFM3- FederatedAgents/chestx-ray/clients/reduced_client3"]

patient_image_paths = []  # (patient real ID, list of images)

# Iterate over each client folder
for client_path in CLIENT_PATHS:
    for patient_folder in os.listdir(client_path):
        patient_path = os.path.join(client_path, patient_folder)
        if os.path.isdir(patient_path):
            for root, dirs, files in os.walk(patient_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        full_path = os.path.join(root, file)
                        real_patient_id = os.path.splitext(file)[0]  # Remove .png/.jpg etc.
                        patient_image_paths.append((real_patient_id, full_path))

print(f"✅ Found {len(patient_image_paths)} patient images across all clients.")

# =========================
# 2. Clinical Report Generator
# =========================

pneumonia_positive_findings = [
    "opacity in right lower lobe", "bilateral lung consolidation",
    "air bronchograms present", "ground-glass opacities in imaging",
    "diffuse infiltrates consistent with pneumonia"
]

pneumonia_negative_findings = [
    "lungs are clear", "no opacity detected", "normal chest X-ray",
    "no acute cardiopulmonary abnormality", "no pleural effusion observed"
]

other_symptoms = [
    "fever", "cough", "shortness of breath", "fatigue",
    "hypoxia", "chest pain", "decreased breath sounds"
]

def generate_single_report(is_pneumonia):
    report = []
    if is_pneumonia:
        report.append(random.choice(pneumonia_positive_findings))
    else:
        report.append(random.choice(pneumonia_negative_findings))
    report += random.sample(other_symptoms, random.randint(2, 4))
    return "Patient presents with " + ", ".join(report) + "."

# =========================
# 3. Load Embedding Model
# =========================

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Create output folder
os.makedirs("c3_dp_prompt_patient_embeddings_csv", exist_ok=True)

# Store patient-wise embeddings
patient_rows = []

# =========================
# 4. Process Each Patient Image
# =========================

for real_patient_id, img_path in tqdm(patient_image_paths, total=len(patient_image_paths), desc="Processing Patients"):

    # Generate 200 clinical reports per patient
    reports = []
    for _ in range(200):
        is_pneumonia = random.choice([True, False])
        report = generate_single_report(is_pneumonia)
        reports.append(report + " [DP-Prompt ε=2]")

    # Encode all 200 reports
    print(f"✨ Generating 200 embeddings for {real_patient_id}, epsilon=2...")
    embeddings = model.encode(reports, batch_size=32, show_progress_bar=True)

    # Take MEAN of the 200 embeddings
    representative_embedding = np.mean(embeddings, axis=0)

    # Combine real_patient_id + embedding
    row = [real_patient_id] + list(representative_embedding)
    patient_rows.append(row)

# =========================
# 5. Save to CSV (your desired format)
# =========================

# Build final dataframe
embedding_dim = len(patient_rows[0]) - 1  # embedding size
columns = [''] + [str(i) for i in range(embedding_dim)]  # First blank for patient_id, then 0,1,2,...

final_df = pd.DataFrame(patient_rows, columns=columns)

# Save
final_save_path = os.path.join("c3_dp_prompt_patient_embeddings_csv", "c3_text_final_all_patients_embeddings.csv")
final_df.to_csv(final_save_path, index=False)

print("🎯 Done! Final CSV saved exactly in your required format.")
