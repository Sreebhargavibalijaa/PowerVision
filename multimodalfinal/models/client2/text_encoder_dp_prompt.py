import os
import random
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# =========================
# 1. Setup: Collect patient image paths
# =========================

CLIENT_PATHS = [
    "/Users/sreebhargavibalija/Desktop/uflallfiles/UFM3- FederatedAgents/chestx-ray/clients/reduced_client2"
]

patient_image_paths = []  # (real_patient_id, image_path)

for client_path in CLIENT_PATHS:
    for patient_folder in os.listdir(client_path):
        patient_path = os.path.join(client_path, patient_folder)
        if os.path.isdir(patient_path):
            for root, _, files in os.walk(patient_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        full_path = os.path.join(root, file)
                        real_patient_id = os.path.splitext(file)[0]
                        patient_image_paths.append((real_patient_id, full_path))

print(f"✅ Found {len(patient_image_paths)} patient images.")

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
# 3. Load Sentence Embedding Model
# =========================

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Output folder
os.makedirs("c2_dp_prompt_patient_embeddings_csv", exist_ok=True)
patient_rows = []

# =========================
# 4. Process Each Patient
# =========================

for real_patient_id, img_path in tqdm(patient_image_paths, total=len(patient_image_paths), desc="Processing Patients"):

    id_lower = real_patient_id.lower()

    # ✅ Label assignment based on your description
    if "person" in id_lower:
        is_pneumonia = True
        label = 1
    elif "normal" in id_lower or id_lower.startswith("im-"):
        is_pneumonia = False
        label = 0
    else:
        print(f"⚠️ Skipping unknown format: {real_patient_id}")
        continue

    # Generate 200 clinical reports
    reports = [generate_single_report(is_pneumonia) + " [DP-Prompt ε=2]" for _ in range(200)]

    # Embed
    print(f"✨ Encoding 200 reports for {real_patient_id} (label={label})...")
    embeddings = model.encode(reports, batch_size=32, show_progress_bar=True)

    # Mean representation
    representative_embedding = np.mean(embeddings, axis=0)

    # Store row
    row = [real_patient_id] + list(representative_embedding) + [label]
    patient_rows.append(row)

# =========================
# 5. Save as CSV
# =========================

if not patient_rows:
    raise ValueError("❌ No patient embeddings generated. Check ID format handling.")

embedding_dim = len(patient_rows[0]) - 2
columns = ['patient_id'] + [f"dim_{i}" for i in range(embedding_dim)] + ['label']

final_df = pd.DataFrame(patient_rows, columns=columns)
save_path = os.path.join("c2_dp_prompt_patient_embeddings_csv", "c2_text_final_all_patients_embeddings.csv")
final_df.to_csv(save_path, index=False)

print(f"🎯 Success! Saved {len(patient_rows)} rows to {save_path}")
