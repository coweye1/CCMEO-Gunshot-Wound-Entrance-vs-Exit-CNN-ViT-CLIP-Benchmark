# CCMEO Gunshot Wound Image Analysis AI Benchmarking: Entrance vs. Exit Wound Classification
Advanced deep learning benchmark study evaluating Convolutional Neural Network-based (CNN), Vision Transformer-based (ViT), and CNN-ViT Hybrid SOTA Architectures on forensic pathology datasets at the Cook County Medical Examiner's Office (CCMEO) with Large-Scale External Validation on the GuWID-UnB Dataset.

---

## 📌 Project Overview

### 💡 Quick & Easy Summary
> 🔍 **"Which image analysis AI model is the smartest at distinguishing a bullet's entrance wound from its exit wound?"**
> 
> This repository presents a comprehensive **AI Performance Exam (Benchmark Study)** that I designed to answer that exact question for forensic pathology. 
> 
> I personally configured, trained, and evaluated 9 established Image AI models using real-world, certified autopsy photography from the **Cook County Medical Examiner's Office (CCMEO)**. Then, to test their true diagnostic adaptability, I challenged them with a completely blind "Final Exam" using the independent **GuWID-UnB dataset** (constructed by Renato Queiroz Nogueira Lira et al., in collaboration with the University of Brasília and other institutions). 

In forensic pathology, distinguishing between **Entrance Wounds** and **Exit Wounds** is a critical task for reconstructing shooting incidents, determining bullet trajectories, and providing medical-legal testimony. 

This repository implements and benchmarks 9 state-of-the-art computer vision architectures, divided into **3 different algorithmic approaches to image recognition**: **Convolutional Neural Networks (CNN)**, **Vision Transformers (ViT)**, and **CNN-ViT Hybrid** models. The goal of this study is to evaluate their capacity to automate and objectively analyze morphology patterns in gunshot wound trauma. Leveraging pure PyTorch and `timm`, all models were evaluated on high-resolution forensic autopsy datasets from the CCMEO and further stress-tested via large-scale external validation to verify real-world algorithmic safety and robustness against severe domain shifts.

---

## 📊 Dataset Specification & Splitting Strategy
The core foundation of this benchmark relies on high-resolution, certified forensic photography meticulously annotated at the Cook County Medical Examiner's Office (CCMEO). 

### 🔍 Cohort Overview
* **Temporal Range:** Data compiled continuously from **2023 to 2026**.
* **Total Enrolled Cohort:** **315 distinct forensic autopsy cases** presenting with firearm trauma.
* **Total Compiled Dataset:** **1,639 high-resolution images**

### ✂️ Manual ROI Extraction & Artifact Elimination (Preventing AI "Cheating")
I spent hours manually cropping every single image into a strict 1:1 square aspect ratio (Region of Interest - ROI) to isolate the immediate wound architecture. During this meticulous manual extraction process, explicit care was taken to strictly exclude all confounding variables from the frame, such as autopsy case number tags, surgical sutures, visible bullets or projectiles lodged near the wound, and non-cutaneous background environments.

> 💡 **Why this process matters — To Prevent AI "Cheating"**
> AI models are notoriously "lazy" copycats. If an autopsy photo contains a case number tag, a specific hospital setting, or surgical sutures, the AI might simply memorize those artificial shortcuts to pass the exam instead of actually evaluating the wound. 
> 
> By completely stripping away these external artifacts and background context, I forced the artificial brain to learn authentic, universal pathological features purely from the morphology of the wound itself.

### 🔄 Data Partitioning Matrix (Strict Case-Independence)
To ensure absolute empirical integrity, the dataset was strictly partitioned at a rigorous case-independent level. This guarantees that all images originating from a single forensic case are restricted entirely to either the Training set or the Validation set, with zero cross-contamination.

> 💡 **Why this partition matters — To Prevent Data Leakage**
> Imagine a single forensic case (Patient A) has 2 different photos taken from slightly different angles or distances. If I randomly split these photos—putting 1 into the Training set and 1 into the Validation set—there is a high risk that the AI will correctly classify Patient A's testing photo for the wrong reasons.
> 
> Instead of genuinely understanding gunshot wounds, the model is highly likely to adapt to Patient A's specific skin tone, body location, or the unique lighting of that specific autopsy room. 
> 
> By locking all images of a single patient onto one side of the fence, I eliminate this bias and ensure a truly honest, rigorous test: the AI is forced to judge a patient it has never, ever seen before.

| Wound Category | Total Images | Training Set | Validation Set |
| :--- | :---: | :---: | :---: |
| **Entrance Wounds** | 979 | 773 | 206 |
| **Exit Wounds** | 660 | 538 | 122 |
| **Combined Total** | **1,639** | **1,311** | **328** |

### ⚖️ Adjusting for Data Imbalance (Weighted Loss)
In this dataset, there are naturally more entrance wound images (979) than exit wound images (660). If left unaddressed, an AI model can easily become biased toward the majority class (entrance wounds), leading to a higher rate of false negatives for exit wounds. 

To ensure completely unbiased and fair diagnostic training, I applied a statistical correction to the loss function (`nn.CrossEntropyLoss`).

> 💡 **Why this mathematical weight matters?**
> Imagine an AI taking a 100-question exam where **80 questions are Entrance Wounds** and **only 20 are Exit Wounds**. A lazy AI could simply guess "Entrance" for every single question without studying at all, and it would still score a deceptive 80%. 
> 
> To balance the scales in this hypothetical exam, we must mathematically make the rarer questions more valuable: since Entrance wounds outnumber Exit wounds by **4 to 1 (80:20)**, misclassifying a rare Exit wound should carry a **4.0x higher penalty**.
> 
> **Applying this to my actual dataset:** > My real-world training set contains **773 Entrance Wounds** and **538 Exit Wounds**. To find the exact fair penalty, I calculated the inverse ratio of the classes:
> 
> $$\text{Penalty Weight for Exit Wounds} = \frac{773 \text{ (Entrance Images)}}{538 \text{ (Exit Images)}} \approx 1.48$$
> 
> By assigning this precise **1.48x higher penalty weight** whenever the model misclassifies a scarcer exit wound, the AI is effectively punished for ignoring the minority class. This actively forces the artificial brain to study the unique, subtle morphological features of both wound types with equal clinical importance.

---

## 🔬 External Validation Cohort (GuWID-UnB Dataset)
To stress-test the domain generalization limits and prevent source-dataset bias, I introduced the completely independent **Gunshot Wound Image Database (GuWID)**. This dataset was constructed as part of the study titled *"Deep Learning-Based Human Gunshot Wounds Classification"* by Renato Queiroz Nogueira Lira et al., in collaboration with the University of Brasília (UnB) and other institutions. The benchmark dataset is officially accessible via their public repository: [pedrogarciafreitas/GuWID-UnB](https://github.com/pedrogarciafreitas/GuWID-UnB).

Serving as a rigorous external "Final Exam," this out-of-distribution (OOD) cohort challenges the models across severe demographic shifts, distinct photography setups, and varied lesion-acquisition protocols.

> 💡 **Why this setup matters?**
> If an AI model trained exclusively on CCMEO data can still successfully classify images from an entirely independent dataset (with different lighting, camera gear, and backgrounds), it proves that the algorithm is not just "cheating" by memorizing site-specific photography styles. Instead, it demonstrates that the AI has truly mastered the universal, authentic pathological features of gunshot wounds. This benchmarking strategy allows me to objectively evaluate and rank how well these models generalize to real-world forensic environments across different international institutions.

| Wound Category (GuWID External Evaluation Cohort) | Total Images |
| :--- | :---: |
| **Entrance Wounds** | **1,883 images** |
| **Exit Wounds** | **671 images** |
| **Combined Total (Robustness Stress-Test)** | **2,554 images** |

* **Scale Contrast:** The external testing cohort (**2,554 images**) is significantly larger than the internal validation subset (**328 images**), providing immense statistical power to evaluate true real-world diagnostic performance and algorithmic clinical safety across distinct institutional frameworks.

---

## 🔬 Evaluated Model Lineup (9 Architectures)
Nine diverse deep learning backbones across three architectural families were trained and optimized using dynamic image resolutions matching their pre-trained infrastructures:

* **CNN Family (Solid Lines):** ResNet50, EfficientNet-B0, ConvNeXt-V2-Tiny
* **ViT Family (Dashed Lines):** ViT-Small, DeiT-Tiny, DINOv2-Small (Self-Supervised)
* **CNN-ViT Hybrid Family (Dotted Lines):** Visformer-Small, CoAtNet-0, MaxViT-Tiny

---

## 📊 Benchmarking Performance Metrics
All 9 vision architectures were trained for a fixed duration of **20 full epochs**. To secure the most robust and clinically optimal checkpoint, the definitive weight files were extracted from the precise training epoch where the framework achieved its **Peak Validation ROC-AUC** score. 

Below are the comparative performance matrices captured under this empirical strategy.

### 📝 Internal Validation (CCMEO Dataset)
Performance quantified via full fine-tuning on the case-independent internal validation partition.

| Rank | Model Name | Model Family | Accuracy | Precision | Recall (Sens.) | F1-Score | **Peak Validation ROC-AUC** | Peak Epoch |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **ConvNeXt-V2-Tiny** | CNN | 0.8354 | 0.7698 | 0.7951 | 0.7823 | **0.8959** | Ep 10 |
| 2 | **DINOv2-Small** | **ViT** | 0.8201 | 0.7890 | 0.7049 | 0.7446 | **0.8821** | Ep 12 |
| 3 | **Visformer-Small** | CNN-ViT Hybrid | 0.8079 | 0.7185 | 0.7951 | 0.7549 | **0.8792** | Ep 6 |
| 4 | CoAtNet-0 | CNN-ViT Hybrid | 0.8049 | 0.7132 | 0.7951 | 0.7519 | 0.8767 | Ep 4 |
| 5 | ViT-Small | **ViT** | 0.7835 | 0.7802 | 0.5820 | 0.6667 | 0.8578 | Ep 12 |
| 6 | MaxViT-Tiny | CNN-ViT Hybrid | 0.8018 | 0.7209 | 0.7623 | 0.7410 | 0.8638 | Ep 11 |
| 7 | DeiT-Tiny | **ViT** | 0.7713 | 0.6478 | 0.8443 | 0.7331 | 0.8532 | Ep 9 |
| 8 | ResNet50 *(Baseline)* | CNN | 0.7
