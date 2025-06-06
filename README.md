<h1 align="center">🧠 PowerVision: Privacy-Preserving Multimodal Embeddings for Federated Learning</h1>

<p align="center">
  <img src="https://img.shields.io/badge/privacy-(ε,δ)DP-blue" />
  <img src="https://img.shields.io/badge/federated-learning-green" />
  <img src="https://img.shields.io/badge/modalities-image--text--tabular-purple" />
  <img src="https://img.shields.io/badge/license-MIT-yellow" />
</p>

> PowerVision enables secure, interpretable, and privacy-preserving multimodal AI at the edge. It fuses **image**, **tabular**, and **text** data into differentially private embeddings for robust federated learning in sensitive domains like healthcare and finance.

---

## ✨ Key Features

✅ Differential Privacy using DP-SGD (per modality)  
✅ Unified Fusion of Chest X-rays, Clinical Notes, and Tabular Vitals  
✅ Federated Learning Support with Client-level Control  
✅ KDE Calibration & Lipschitz Regularization for Robustness  
✅ Built-in Interpretability using Neural Additive Models (NAMs)  
✅ Open-source, Reproducible, Lightweight

---

## 🧩 System Architecture

```
+--------------------+      +------------------+
| X-ray (CNN + DP)   |      | Clinical Notes   |
|                    |----> | (Transformer + DP)|  
+--------------------+      +------------------+
                                  |
+--------------------+           |
| Vitals (MLP + DP)  |-----------+
+--------------------+

          |
          v
+-------------------------+
|   Fused Representation  |
+-------------------------+
          |
          v
+-------------------------+
| Federated PowerVision   |
|        Model            |
+-------------------------+
          |
          v
+-------------------------+
| Explainability Module  |
|        (NAMs)          |
+-------------------------+
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-org/powervision.git
cd powervision
pip install -r requirements.txt
```

---

## 🚀 Run PowerVision

```bash
python run_powervision.py --modality image text tabular --dp --federated --explain
```

### Flags:
- `--dp` : Enable differential privacy
- `--federated` : Run in a federated setting
- `--explain` : Enable interpretable feature analysis (NAM/SHAP)

---

## 🧪 Performance Snapshot

| Dataset     | ROC-AUC | Log Loss | ε (DP) | Fairness Gap |
|-------------|---------|----------|--------|---------------|
| ChestX-ray  | 0.91    | 0.32     | 1.5    | 0.03          |
| MIMIC-III   | 0.88    | 0.35     | 2.1    | 0.05          |

---

## 🧠 Interpretability

PowerVision supports **Neural Additive Models** at the client level.  
Top contributing pixels, vitals, or tokens are visualized interactively.

---

## 🔒 Privacy Stack

- **DP-SGD** for each modality
- **Power-Learning** privacy embeddings
- **Lipschitz constraints** for robustness
- **Uncertainty Calibration** using KDE

---

## 🗂️ Project Structure

```
powervision/
├── data/                   # Preprocessed multimodal datasets
├── models/                 # LLMs, CNNs, MLPs
├── privacy/               # DP and Power-Learning utils
├── fusion/                 # Fusion and calibration logic
├── explainability/         # NAM, SHAP modules
├── run_powervision.py      # Main runner script
└── README.md
```

---

## 🤝 Contributors

- 👩‍🔬 **Sree Bhargavi Balija** — Federated Learning, Privacy, and Multimodal Fusion  
- 👨‍💻 [Add Collaborator Name] — Backend Optimization and Deployment

---

## 📄 License

This project is licensed under the MIT License © 2025 PowerVision Contributors

---

<p align="center"><b>Transforming edge intelligence with privacy and interpretability 🔐</b></p>
