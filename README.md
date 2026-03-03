# 🎵 Music Genre Classification – Data Mining Project

## 📌 Project Overview

This project analyzes and compares music tracks using fundamental data mining techniques. The objective is to preprocess structured audio features, perform feature extraction, visualize feature distributions, and compute similarity between songs.

The project was implemented using Python and standard data mining libraries as part of an academic Data Mining assignment.

---

## 📊 Dataset

- **Dataset Used:** GTZAN Music Genre Dataset  
- **Source:** Kaggle  
- **Total Tracks:** 1000  
- **Genres:** 10 (Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock)  
- **Numerical Features:** 57 audio features per track  
- Each track is 30 seconds long and represented as structured numerical feature vectors.

---

## ⚙️ Techniques Applied

### 1️⃣ Data Preprocessing

- Checked for missing and duplicate values  
- Removed non-informative columns (`filename`, `length`, `label`)  
- Applied **Z-Score Standardization**

Z = (X - μ) / σ

Where:  
- X = original feature value  
- μ = mean of the feature  
- σ = standard deviation  

- Applied **Principal Component Analysis (PCA)**  
  - Reduced 57 features to 23 components  
  - Retained over 90% of total variance  

---

### 2️⃣ Feature Extraction

- Implemented **Spectral Contrast** extraction using `librosa`  
- Extracted 7 frequency band contrast features  
- Computed mean and variance to create fixed-length feature vectors  
- Demonstrated feature extraction on raw audio files  

---

### 3️⃣ Data Visualization

- Used **Box Plot (Box-and-Whisker Plot)**  
- Compared feature distributions before and after standardization  
- Verified correction of scale imbalance visually  

---

### 4️⃣ Similarity Analysis

- Applied **Stratified Sampling** (1 song per genre → 10 songs total)  
- Used **Cosine Similarity** on standardized features  
- Computed a 10 × 10 pairwise similarity matrix  
- Identified the pair with maximum similarity  

---

## 🔎 Key Results

- PCA reduced dimensionality from **57 → 23 features** while retaining **90.32% variance**  
- Standardization successfully corrected scale imbalance  
- Most similar pair identified:

  **Blues & Rock**  
  Cosine Similarity Score ≈ **0.812**

This indicates strong acoustic similarity between the two genres.

---
## 📎 Kaggle Notebook Links

You can view the implementation and execution results on Kaggle:

- [T2](https://www.kaggle.com/code/twisterteja/task-2-final)
- [T3](https://www.kaggle.com/code/twisterteja/task-3-final)
- [T4](https://www.kaggle.com/code/twisterteja/task-4-final)
- [T5](https://www.kaggle.com/code/twisterteja/task-5-final)

---
## 🛠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Librosa  
- Matplotlib  

---

---

## 🚀 Conclusion

This project demonstrates how preprocessing, feature extraction, dimensionality reduction, visualization, and similarity measures can be systematically applied to real-world audio datasets.

The complete pipeline ensures robust data preparation and meaningful similarity computation across music genres.

