### 📌 Customer Sentiment Analysis — Machine Learning Project

### Course: Machine Learning (CS 135)
### Author: Maida Raza

### 📝 Overview

This project analyzed 2,400 single-sentence customer product reviews from Amazon, Yelp, and IMDB to classify sentiment (positive or negative) using three machine learning models:

* Support Vector Machines (SVM)

* Random Forests (RF)

* Multi-Layer Perceptrons (MLP)

The goal was to build models capable of automatically assigning sentiment labels to new customer reviews, streamlining the feedback assessment process.

### 📂 Dataset

* Sources: Amazon, Yelp, IMDB

* Size: 2,400 reviews (balanced binary labels)

* Labels: 1 = Positive, 0 = Negative

### 🔧 Data Preprocessing

* Text cleaning:
  - Removed numbers, punctuation, non-ASCII characters
  - Lowercased text, normalized whitespace
  - Feature engineering:
      * TF-IDF (top 1,000 words) to capture importance of terms
      * Bag-of-Words (BoW) with binary presence/absence encoding

### 📊 Models & Methods

* SVM: RBF kernel; tuned C and gamma across 5-fold stratified CV

* Random Forest: Varied n_estimators and max_depth across 5-fold stratified CV

* MLP: Tested varying hidden layers across 5-fold stratified CV

### 📈 Key Findings

* Best SVM performance at moderate gamma (0.01–0.1) and C between 1–10

* Random Forest performance improved with depth until overfitting appeared

* MLP showed sensitivity to both depths upto a point

### 📦 Tech Stack

* Python: scikit-learn, keras, tensorflow, pandas, numpy, matplotlib

* ML Methods: TF-IDF, BoW, SVM, Random Forest, MLP

* Validation: 5-fold Stratified Cross-Validation

