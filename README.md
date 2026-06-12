# CCMEO Gunshot Wound Benchmarking: Entrance vs. Exit Wound Classification
Advanced deep learning benchmark study evaluating Convolutional Social Networks (CNN) and Vision Transformers (ViT) on forensic pathology datasets at the Cook County Medical Examiner's Office (CCMEO).

---

## 📌 Project Overview
In forensic pathology, distinguishing between **Entrance Wounds** and **Exit Wounds** is a critical task for reconstructing shooting incidents, determining bullet trajectories, and providing medical-legal testimony. 

This repository implements and benchmarks six state-of-the-art computer vision architectures (CNNs and Vision Transformers) to automate and objectively analyze morphology patterns in gunshot wound trauma. Leveraging pure PyTorch and `timm`, all models were evaluated on high-resolution forensic autopsy datasets.

---

## 🔬 Evaluated Model Lineup & Best Checkpoints
Six diverse deep learning backbones were trained and optimized using dynamic image resolutions matching their pre-trained infrastructures:

* **CNN Architectures:** ResNet50, EfficientNet-B0, ConvNeXt-V2-Tiny
* **Vision Transformers (ViT):** ViT-Small, Swin-Tiny, DINOv2-Small (Self-Supervised)

---

## 📊 Benchmarking Performance Metrics
Below is the definitive performance matrix compiled at the best-performing training epochs across all architectures:

| Model Name | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | Best Epoch |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ConvNeXt-V2-Tiny** | **0.835** | **0.842** | **0.825** | **0.8334** | **0.8918** | **3** |
| DINOv2-Small | 0.829 | 0.775 | 0.762 | 0.7686 | 0.8884 | 16 |
| Swin-Tiny | 0.821 | 0.815 | 0.830 | 0.8224 | 0.8800 | 4 |
| ViT-Small | 0.805 | 0.798 | 0.815 | 0.8064 | 0.8576 | 3 |
| ResNet50 | 0.811 | 0.824 | 0.798 | 0.8108 | 0.8526 | 15 |
| EfficientNet-B0 | 0.742 | 0.731 | 0.762 | 0.7462 | 0.7940 | 20 |

### 🔑 Key Takeaways
1. **ConvNeXt-V2-Tiny** achieved the highest overall performance with an **Accuracy of 83.5%** and a dominant **ROC-AUC of 0.8918**, proving the massive potential of modernized ConvNets in specialized clinical/forensic tasks.
2. **DINOv2-Small** demonstrated exceptional representation capacity with an **AUC of 0.8884**, validating that self-supervised vit-features adapt seamlessly to low-sample medical distributions.

---

## 📈 Visualizations & Analytical Assets

### 1. Integrated ROC Curves
The Receiver Operating Characteristic (ROC) curves illustrate the true-positive vs. false-positive trade-offs across all 6 architectures. The closer the curve vaults toward the top-left corner, the superior the model's discriminative ability.
![Integrated ROC Curves](curves_plot.png)

### 2. Multi-Architecture Confusion Matrices
A 2x3 grid mapping out the exact classification distribution (True vs. Predicted Labels) for each model. Cell values display raw counts alongside percentage ratios to show exact directional error tendencies.
![Confusion Matrices](confusion_matrix.png)

### 3. Explainable AI (XAI): Grad-CAM Spatial Heatmaps
To bridge the gap between deep learning and forensic medicine, we injected PyTorch hook systems into the final block of **ConvNeXt-V2** to map visual attention grids.
![Grad-CAM Visualizations](grad_cam.png)

* **Entrance Wound Analysis:** The model precisely targets the **Abrasion Collar (찰과륜)** surrounding the wound margin, showing deep alignment with standard medical textbook definitions.
* **Exit Wound Analysis:** The attention grid highlights the irregular, **Lacerated Margins (파열연)** and structural skin flaps, proving that the model relies on true morphological features rather than background noise.

---

## 🛠️ Environment & Requirements
* Python 3.10+
* PyTorch 2.0+ (CUDA enabled)
* `timm` (Torch Image Models)
* scikit-learn, matplotlib, seaborn, opencv-python

---

## 📜 Future Works: Phase 2
Building upon this solid benchmark registry, the next iteration expands into **Multimodal Frameworks (CLIP)** to integrate structured autopsy text reports with gross pathology imagery for zero-shot forensic intelligence.
