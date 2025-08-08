# Rainfall Prediction using Machine Learning

## Overview
This project develops a machine learning model to predict whether it will rain on a given day based on atmospheric conditions such as temperature, humidity, pressure, cloud cover, wind speed, and wind direction.  
The main objective is to use historical weather data to train classification models that can make accurate predictions, even when the dataset is imbalanced.

This work is part of the **50 Days, 50 Projects – Machine Learning Challenge**.

## Objectives
- Build a classification model to predict rainfall occurrence.
- Perform exploratory data analysis (EDA) to identify key weather patterns.
- Handle missing values and imbalanced datasets.
- Compare the performance of multiple classification algorithms.

## Dataset
- Source: `Rainfall.csv`
- Total rows: 366
- Features include:
  - Pressure
  - Maximum, Minimum, and Average Temperatures
  - Dew Point
  - Humidity
  - Cloud Cover
  - Sunshine Hours
  - Wind Direction
  - Wind Speed
  - Rainfall (Target: Yes/No)

## Libraries Used
- **NumPy**, **Pandas**
- **Matplotlib**, **Seaborn**
- **scikit-learn** (Logistic Regression, SVC, preprocessing, metrics)
- **XGBoost** (XGBClassifier)
- **imblearn** (RandomOverSampler)

## Project Workflow

### 1. Data Loading and Inspection
- Loaded the dataset and checked data types, null values, and summary statistics.
- Identified spaces in column names and corrected them.
- Filled missing values using the mean for numeric features.

### 2. Data Preprocessing
- Converted categorical target variable (`yes`/`no`) into numeric (`1`/`0`).
- Checked for highly correlated features and dropped redundant ones (`maxtemp`, `mintemp`).
- Applied **Random Over Sampling** to balance the target variable classes.
- Normalized numerical features using **StandardScaler**.

### 3. Exploratory Data Analysis (EDA)
- Visualized rainfall distribution using a pie chart.
- Compared average feature values for rainy vs non-rainy days.
- Plotted feature distributions and boxplots to detect skewness and outliers.
- Generated a correlation heatmap to detect multicollinearity.

### 4. Model Training
Trained and evaluated three classification models:
1. **Logistic Regression**
2. **XGBoost Classifier**
3. **Support Vector Classifier (RBF kernel)**

### 5. Model Evaluation
- Metrics: ROC AUC Score, Confusion Matrix, Classification Report.
- Results:
  - Logistic Regression: Train AUC = 0.889, Validation AUC = 0.897
  - XGBClassifier: Train AUC = 0.990, Validation AUC = 0.841 (overfitting observed)
  - SVC: Train AUC = 0.903, Validation AUC = 0.886

The **SVC** and **Logistic Regression** models showed the best generalization performance.

### 6. Final Evaluation
- Plotted confusion matrix for the SVC model.
- Generated precision, recall, and F1-score for both classes.


