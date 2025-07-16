# Credit Approval Prediction

This project explores how machine learning can support credit decision-making by predicting whether a credit application should be approved or denied. The analysis addresses common real-world challenges, including missing values, mixed data types, class imbalance, and overfitting. Several classification models are trained and evaluated to identify the most effective approach.

The code is written in Python and presented in a Jupyter notebook to promote clarity, reproducibility, and modular experimentation.

## Overview

Automating credit approval decisions is a common use case for predictive analytics in financial institutions. In this project, we implement a complete machine learning pipeline that:

- Cleans and preprocesses structured tabular data
- Encodes nominal and ordinal categorical variables
- Balances skewed class distributions
- Trains multiple supervised classification models
- Evaluates models with robust validation strategies
- Identifies and mitigates overfitting

The objective is to assess which algorithms provide the most accurate and generalizable performance when making binary credit decisions.

## Key Features

- End-to-end data preparation and modeling pipeline
- Custom imputation strategies for missing data
- Encoding for both nominal and ordinal variables
- Cross-validation with 5, 7, and 10 folds
- Comparison of seven supervised learning models
- Techniques to handle class imbalance (e.g., SMOTE)
- Evaluation using multiple performance metrics

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/MinaaBobinaa/credit-approval-prediction.git
cd credit-approval-prediction
```


### 2. Set Up a Virtual Environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3.Install Required Packages

```bash
pip install -r requirements.txt
```
If requirements.txt is not available, install manually:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

### 4.Launch the Jupyter Notebook
```bash
jupyter notebook
```

## 5.Technical Highlights
- Encoding with `OneHotEncoder` and `OrdinalEncoder`
- Feature scaling using `StandardScaler`
- Class balancing using `RandomOverSampler` and `SMOTE` from `imbalanced-learn`
- Training and evaluation of the following classification models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors
  - Support Vector Machine
  - Naive Bayes
  - Gradient Boosting
- Validation techniques:
  - 70/30 train-test split
  - k-fold cross-validation (k=5, 7, and 10)
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC AUC
- Visualization tools:
  - Confusion matrices
  - Classification reports
  - Correlation heatmaps and feature distributions
- Overfitting diagnosis by comparing training vs. validation scores

## 7. Purpose

This project was developed for academic purposes to demonstrate the application of machine learning to credit risk analysis. It highlights the importance of clean data pipelines, interpretable modeling, and rigorous evaluation when building decision support systems.

## 8. License

This repository is intended for academic use only. Redistribution or commercial use is not permitted without explicit permission.