# Customer Sentiment Analysis

## 📌 Overview
This project classifies customer sentiment (positive vs. negative) from product reviews collected across **Amazon, Yelp, and IMDB**.  
The goal was to build machine learning models that can automatically assign sentiment labels to new reviews.

---

## 📊 Dataset
- **2,400 single-sentence reviews** (balanced between positive and negative).  
- Sources: `amazon.com`, `yelp.com`, `imdb.com`.  
- Additional **600 unseen test reviews** used for final evaluation.  

---

## 🛠 Preprocessing
- Removed numbers, punctuation, and non-English characters.  
- Converted text to lowercase, normalized whitespace.  
- Feature representations:
  - **TF-IDF** (top 1,000 informative words).  
  - **Bag-of-Words (BoW)** (binary presence/absence of top 1,000 words).  

---

## 🤖 Models
Implemented and compared:
1. **Support Vector Machines (SVM)** (RBF kernel, tuned `C` and `γ`)  
2. **Random Forests** (tuned `n_estimators`, `max_depth`)  
3. **Multi-Layer Perceptrons (MLP)** (varied hidden layers and learning rate, trained with Adam optimizer)

---

## 🔎 Key Findings
- **SVM**: Best performance with moderate `C` (1–10) and moderate `γ` (0.01–0.1). Too large/small values caused underfitting or overfitting.  
- **Random Forests**: Depths of 5–15 with 100–200 trees gave best bias–variance trade-off. Deeper/larger forests risked overfitting.  
- **MLP**:  
  - Validation performance plateaued after 2 layers.  
  - Learning rate **1e-3** yielded the best generalization.  
  - Achieved lowest error overall.

---

## 🏆 Best Model
- **Multi-Layer Perceptron (MLP)**  
- Training error ≈ 0, validation error ≈ 0.2  
- **100% accuracy** on the 600 test reviews  

---

## ⚙️ Tech Stack & Libraries
- **Programming Language**: Python 
- **Core Libraries**:
  - `scikit-learn` → preprocessing (TF-IDF, BoW), SVM, Random Forests, cross-validation  
  - `tensorflow.keras` → MLP model building, training, evaluation  
  - `numpy` / `pandas` → data wrangling and matrix manipulation  
  - `matplotlib` → plotting and visualization  

---

## 🚀 Next Steps
- Scale model to larger datasets.  
- Explore word embeddings (e.g., Word2Vec, GloVe, BERT) for richer feature representation.  
- Apply to multilingual datasets for broader applicability.   
