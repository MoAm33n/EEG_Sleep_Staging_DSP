#  EEG Sleep Stage Classification: A DSP and Machine Learning Approach

## Overview
This repository contains the code and resources for an advanced Signals and Systems project focused on **Automatic Sleep Stage Classification** using Polysomnography (PSG) data. The goal is to design and implement a robust Digital Signal Processing (DSP) pipeline that extracts meaningful features (e.g., spectral power bands) from raw EEG signals to accurately predict the five standard sleep stages.

### Core Problem Statement
The project addresses the challenge of automating the labor-intensive process of human sleep scoring (hypnogram generation). We aim to classify 30-second EEG epochs into one of five sleep stages (Wake, N1, N2, N3, REM) based solely on physiological signal characteristics.

---

## 🛠️ Technology Stack
| Component | Tool/Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.x | Main development environment. |
| **Data Handling** | **MNE-Python** | Loading, handling, and segmenting complex EDF/PSG files and annotations. |
| **DSP/Filtering** | **SciPy / NumPy** | Designing and implementing IIR (Butterworth) and Notch filters, and calculating FFT/PSD. |
| **Machine Learning** | **Scikit-learn** | Training classifiers (Random Forest, SVM) and evaluating performance. |
| **Version Control** | **Git / GitHub** | Collaboration and version tracking. |

---

## 📑 Project Phases (DSP Pipeline)

The project is structured into three main phases, reflecting a standard signal processing and machine learning workflow:

### Phase 1: Data Preprocessing and Alignment
* **Data Source:** **Sleep-EDF (Expanded) Database** (PhysioNet).
* **Task:** Load continuous raw PSG data (`*-PSG.edf`) and align it with the expert labels (`*-Hypnogram.edf`).
* **Output:** A list of **30-second epochs**, each perfectly labeled with its corresponding sleep stage (W, N1, N2, N3, R).

### Phase 2: Signal Filtering and Cleaning (S&S Core)
* **Goal:** Isolate the target brain rhythms (Alpha, Delta, Theta, etc.) and remove artifacts.
* **Techniques:**
    * **High-Pass Filter (e.g., > 0.3 Hz):** Remove baseline drift.
    * **Notch Filter (e.g., 50 Hz):** Remove power line interference.
    * **Bandpass Filter:** Isolate the main $\sim 0.5 \text{Hz}$ to $\sim 40 \text{Hz}$ spectrum.

### Phase 3: Feature Extraction and Classification
* **Feature Extraction (DSP Focus):** Calculate **Relative Power Spectral Density (PSD)** features (e.g., $\text{Power}_{\text{Delta}} / \text{Power}_{\text{Total}}$) for each epoch. This translates the time-series signal into a compact feature vector.
* **Classification (ML):** Train a **Random Forest** or **SVM** model to predict the sleep stage based on the extracted feature vector.
* **Evaluation:** Measure performance using **Accuracy** and the **Macro-Averaged F1 Score** (critical for handling class imbalance).

---

## 🏃 Getting Started

### Prerequisites
1.  Clone this repository: https://github.com/MoAm33n/EEG_Sleep_Staging_DSP.git 
    ```bash
