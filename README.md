# 🎵 Music Genre Classification – Data Mining Project

---

## 📌 Project Overview

This project focuses on applying data mining and machine learning techniques for **music genre classification** using the GTZAN dataset.

The work is divided into two phases:

- **Phase 1 (Assignment 1):**
  - Data preprocessing  
  - Feature extraction  
  - Visualization  
  - Similarity analysis  

- **Phase 2 (Assignment 2):**
  - Feature engineering  
  - Model building  
  - Hyperparameter tuning  
  - Performance evaluation  

The complete pipeline demonstrates how raw audio features can be transformed into meaningful insights and accurate classification models.

---

## 📊 Dataset

- **Dataset:** GTZAN Music Genre Dataset  
- **Source:** Kaggle  
- **Total Tracks:** 1000  
- **Genres:** 10  
  *(Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock)*  
- **Features:** 57 numerical audio features per track  
- **Duration:** 30 seconds per track  

Each track is represented as a structured numerical feature vector.

---

## Phase 1 – Data Mining Tasks

### 1️⃣ Data Preprocessing
- Checked for missing and duplicate values  
- Removed non-informative columns (`filename`, `length`)  
- Applied **Z-score standardization**:

Z = (X - μ) / σ

* Ensured all features contribute equally  

---

### 2️⃣ Feature Extraction

* Implemented **Spectral Contrast** using `librosa`  
* Extracted 7 frequency band features  
* Computed mean and variance → 14 new features  

---

### 3️⃣ Data Visualization

* Used **Box Plots**  
* Compared distributions before and after scaling  
* Verified correction of scale imbalance  

---

### 4️⃣ Similarity Analysis

* Used **Stratified Sampling** (1 sample per genre)  
* Applied **Cosine Similarity**  
* Generated a 10 × 10 similarity matrix  

#### 🔎 Key Insight:

* Most similar pair: **Blues & Rock**  
* Similarity score ≈ **0.81**  

---

## 🤖 Phase 2 – Classification & Evaluation

### 1️⃣ Preprocessing

* Standardization using `StandardScaler`  
* Label encoding of target variable  
* Train-test split (80% – 20%, stratified)  

---

### 2️⃣ Feature Extraction Methods

#### 🔹 PCA (Principal Component Analysis)

* Reduced 57 features → **~23 components**  
* Retained **~90% variance**  

#### 🔹 SelectKBest (ANOVA F-test)

* Selected top **k features (10–40)**  
* Best k determined using GridSearch  

---

### 3️⃣ Classification Models

* Decision Tree  
* K-Nearest Neighbors (KNN)  
* Naive Bayes  

---

### 4️⃣ Hyperparameter Tuning

* Used **GridSearchCV**  
* 5-fold **Stratified Cross-Validation**  

---

## 📊 Results Summary

| Feature Method | Model         | Test Accuracy    |
|----------------|--------------|------------------|
| PCA            | KNN          | 70.5%            |
| SelectKBest    | KNN          | **71.0% (BEST)** |
| PCA            | Naive Bayes  | 56.0%            |
| SelectKBest    | Naive Bayes  | 56.5%            |
| PCA            | Decision Tree| 50.0%            |
| SelectKBest    | Decision Tree| 52.5%            |

---

## 🏆 Best Model

* **Model:** K-Nearest Neighbors (KNN)  
* **Feature Method:** SelectKBest  
* **Accuracy:** **71.0%**  

### 🔧 Best Parameters:

* `n_neighbors = 5`  
* `metric = manhattan`  
* `weights = distance`  

---

## 📈 Key Insights

* KNN performs best due to **distance-based similarity learning**  
* SelectKBest slightly outperforms PCA  
* PCA reduces dimensionality but may lose some discriminative detail  
* Decision Trees perform poorly on continuous audio data  
* Naive Bayes is fast but limited by independence assumptions  

---

## 📎 Kaggle Notebook Links

*  [Task 2 (Preprocessing) ](https://www.kaggle.com/code/twisterteja/task-2-final)
*  [Task 3 (Feature Extraction)](https://www.kaggle.com/code/twisterteja/task-3-final) 
*  [Task 4 (Visualization)](https://www.kaggle.com/code/twisterteja/task-4-final)  
*  [Task 5 (Similarity)](https://www.kaggle.com/code/twisterteja/task-5-final) 
*  [Assignment 2 (Final Classification Notebook) ](https://www.kaggle.com/code/twisterteja/dm-assignment2-teja)

---

## 🛠 Technologies Used

* Python  
* Pandas  
* NumPy  
* Scikit-learn  
* Librosa  
* Matplotlib  

---

## 📂 Submitted Files

* `CODE_4.zip` → Python code / notebook  
* `Eval_4.pdf` → Evaluation report  
* `README.pdf` → Project details  

---

## 🚀 Conclusion

This project demonstrates a complete data mining pipeline from preprocessing to model evaluation. The results show that **KNN combined with SelectKBest feature selection** is the most effective approach for music genre classification on the GTZAN dataset.

The study highlights the importance of:

* Feature engineering  
* Model selection  
* Hyperparameter tuning  

---
