# Sleep Stage Identification Using EEG Band-Power Features  
### Signals & Systems Project – 2025  
**Author:** Mohammed Am  
**Repository:** https://github.com/MoAm33n/EEG_Sleep_Staging_DSP

---

## 1. Introduction

Sleep staging is the task of classifying 30-second EEG segments into standard sleep stages:
- **W** – Wake  
- **N1** – Light sleep  
- **N2** – Intermediate sleep  
- **N3** – Deep sleep  
- **R** – REM sleep  

In this project, we implement a complete **signals and systems–based sleep stage classifier** using:
- Time–frequency analysis
- Band-power feature extraction
- Classical machine learning

The goal is to understand how signal processing concepts—filtering, Fourier analysis, frequency-domain interpretation—can be used for a real biomedical classification task.

All code is publicly available at:  
➡️ **https://github.com/MoAm33n/EEG_Sleep_Staging_DSP**

---

## 2. Dataset

We use the **Sleep-EDF Expanded dataset**, which contains overnight EEG recordings sampled at 100 Hz with manually annotated sleep stages.

Preprocessing steps:
1. Resampling to a uniform rate (if needed)
2. Normalizing amplitude
3. Splitting into **30-second epochs**
4. Mapping sleep labels to the 5-stage scheme: W, N1, N2, N3, R

---

## 3. Methodology

### 3.1 Signal Filtering
Each 30-second epoch is filtered into standard EEG clinical bands using band-pass FIR/IIR filters:

| Band | Range |
|------|--------|
| Delta | 0.5–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–12 Hz |
| Sigma | 12–15 Hz |
| Beta  | 15–30 Hz |

Filtering demonstrates key Signals & Systems concepts:
- LTI filtering  
- Frequency response  
- Band-limited energy extraction  

### 3.2 Band-Power Feature Extraction  
For each filtered signal, we compute its **band power**, defined as:

\[
P = \frac{1}{N} \sum_{n=0}^{N-1} x^2[n]
\]

This approximates the signal energy in each frequency band and provides a 5-dimensional feature vector per epoch.

### 3.3 Feature Normalization  
All feature vectors are standardized using:

\[
z = \frac{x - \mu}{\sigma}
\]

### 3.4 Classifier  
We train a classical ML model (Random Forest / SVM depending on experiment variant in the repo).  
No deep learning is used — the focus is on DSP-driven features.

### 3.5 Evaluation Protocol  
We use **subject-wise testing**, meaning:
- Train on some subjects  
- Test on completely unseen subjects  

This makes evaluation realistic and avoids data leakage.

---

## 4. Results

### 4.1 Confusion Matrix

![Confusion Matrix](./Figure_2.png)

**Interpretation:**

- **Wake (W)** is recognized extremely well (7390 correct).  
- **N2** also has high accuracy (1716 correct), which is expected because N2 has a clear sigma/alpha pattern.
- **N3** is moderately good (497 correct), matching the strong delta dominance.
- **R** classification is reasonable (617 correct), though some confusion with N1/N2 exists.
- **N1** is the weakest stage (common in all sleep staging literature).  
  It is often confused with:
  - N2 (55)
  - R (79)

This aligns with known physiological ambiguity between N1 and REM.

---

## 5. Discussion

### Strengths
- Fully explainable, rule-based DSP pipeline  
- High accuracy on major stages (W, N2, N3)  
- Lightweight model suitable for embedded devices  
- Strong subject-wise generalization

### Limitations
- N1 remains difficult with only band-power features  
- Single EEG channel limits feature richness  
- No temporal modeling (sleep stages are sequentially dependent)

### Potential Improvements
1. Add more features:  
   - Spectral entropy  
   - Hjorth parameters  
   - Zero-crossing rate  
2. Use multi-channel EEG  
3. Add context with Hidden Markov Models  
4. Transition from machine learning → deep learning (CNNs/transformers)

---

## 6. Signals & Systems Concepts Demonstrated

This project integrates key S&S topics:

### ✔ Filtering  
Band-pass FIR/IIR filters are used to isolate clinical EEG bands.

### ✔ Fourier Transform  
Power spectral density relates directly to squared FFT coefficients.

### ✔ Energy of a signal  
Band power approximates the energy in frequency-limited components.

### ✔ Linear Systems  
The EEG → filter → energy pipeline demonstrates an LTI system.

### ✔ System identification  
Different sleep stages can be viewed as different “states” generating different frequency content.

---

## 7. Conclusion

This project demonstrates how classical signals and systems tools can be used to classify sleep stages from EEG.  
Using simple **band-power features** and a **classical ML classifier**, we achieve strong performance on W, N2, N3, and R, with expected difficulty on N1.

The project shows that even without deep learning, **DSP alone can extract meaningful physiological patterns** from biomedical signals.

---

## 8. Repository

All code is available at:

➡️ **https://github.com/MoAm33n/EEG_Sleep_Staging_DSP**

Please refer to  `src/` for full implementation details.

