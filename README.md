# 🔧 Hyperparameter Tuning on Classification Case

This repository contains a notebook that demonstrates how to perform **hyperparameter tuning** on various classification models using a binary classification dataset. The goal is to evaluate and improve model performance through systematic tuning using Scikit-Learn's `GridSearchCV`.

---

## 📚 Dataset

- The dataset used involves a binary classification task (https://drive.google.com/file/d/1LVIEzGbnuvHUKPVCZpifk_fPelQ6Wdy-/view?usp=sharing)
- The target variable contains two classes: `0` and `1`.
- It is assumed the dataset is pre-cleaned: no missing values, no categorical encoding required.

---

## 📌 Objectives

- Train and evaluate baseline classification models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost

- Perform hyperparameter tuning using `GridSearchCV` for fine-tuning.

- Evaluate model performance using Recall score.

---

## ⚙️ Analysis Workflow

1. **Data Preparation**
   - Split data into training and testing sets (80:20)
   - Apply feature scaling using `StandardScaler` where needed

2. **Baseline Modeling**
   - Train and evaluate each model with default hyperparameters
   - Select the top-performing model for tuning

3. **Hyperparameter Tuning**
   - Conduct randomized search for initial tuning
   - Follow with grid search for best hyperparameter combination

4. **Final Evaluation**
   - Evaluate the tuned model using metrics and visualization
   - Compare results with the baseline model

---

## 📊 Key Results

- The models showed significant improvement after tuning (better Recall Score).
- The models without tuning exhibit signs of overfitting, as seen in the high recall on the training set, which drops significantly when applied to the validation and test sets.

---

## 🧰 Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
```

If you have any suggestions or feedback, please don't hesitate to contact to me in direct message on Email or LinkedIn: noviaanggita047@gmail.com or https://www.linkedin.com/in/novia-anggita-aprilianti/




#MachineLearning #DataScience #HyperparameterTuning #Classification #ChurnAnalysis #ScikitLearn #Python #ModelOptimization #MLPortfolio #GitHubProject

