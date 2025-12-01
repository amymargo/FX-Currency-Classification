# FX Currency Classification Model

This project builds a machine learning model that predicts which currency (EUR, AUD, USD) a foreign exchange quote belongs to using only the **price** and the **timestamp** of the quote.  
The notebook includes full data exploration, feature engineering, model training, hyperparameter tuning, and a production-style prediction function.
The final model is able to classify currencies with **over 98% accuracy.**

---

## Project Overview

Foreign exchange datasets often contain multiple currencies with similar price magnitudes.  
This project classifies each quote using:

- Raw price level  
- Normalized price (Z-score)  
- Time-of-day behavior  
- Day-of-week  
- Trading session indicators (Asia / EU / US)

The final model uses a **GradientBoostingClassifier**, chosen for its strong performance on nonlinear signals and medium-sized datasets.

---

## Exploratory Data Analysis (EDA)

The notebook includes visualizations to highlight the structure in the data:

### 1. Price Distribution by Currency
Shows distinct price ranges for EUR, AUD, and USD, with EUR tightly clustered, AUD appearing at higher price levels, and USD in the middle.

### 2. Time-Series Price Movements
All three currencies follow similar long-term trends but maintain consistent separation in their absolute price levels throughout the year.

### 3. 3. Return Behavior
Simple returns and log returns show:
- AUD has the largest variability
- EUR is the most stable
- USD sits in the middle

### 4. Hour-of-Day and Global Session Patterns
Strong time-based signals:
- AUD appears mostly during EU daytime hours
- EUR appears heavily during late-night + early-morning
- USD concentrates later in the day

---

## Feature Engineering

The model uses the following engineered features:

- `PRICE`  
- `PRICE_Z` (normalized price)  
- `RETURNS` (percent change)  
- `LOG_RETURN`  
- `HOUR`  
- `DAY_OF_WEEK`  
- `SESSION_ASIA`, `SESSION_EU`, `SESSION_US` (one-hot)

Rolling means and rolling volatility were removed because they cannot be computed for arbitrary isolated timestamps (e.g., future predictions).
The final model only uses features that can be computed from a single input.

---

## Model Training

- Split the data using `train_test_split` with `stratify=y`.
- Trained a Gradient Boosting model with the following tuned hyperparameters:
  - `n_estimators=300`  
  - `learning_rate=0.05`  
  - `max_depth=2`

Validation accuracy: **~98.77%**

---

## Hyperparameter Tuning

Grid search was used to test combinations of:
- Learning rates: 0.02, 0.05, 0.1  
- Number of trees: 150, 300, 500  
- Depths: 2, 3, 4  

---

## Prediction Function

A production-style function `predict_currency(price, timestamp)` accepts:

- Any price value  
- Any timestamp (**past or future**)  

It constructs all required features and returns:

- Predicted currency  
- Prediction probability  

---
