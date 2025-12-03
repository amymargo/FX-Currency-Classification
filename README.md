# FX Currency Classification Model

This project builds a machine learning model that predicts which currency (EUR, AUD, USD) a foreign exchange quote belongs to using only the **price** and the **timestamp** of the quote.

The notebook includes full data exploration, feature engineering, model training, hyperparameter tuning, and a prediction function designed for real-time deployment, allowing the model to classify any future quote.

Foreign exchange datasets contain multiple currencies with similar price structures. The final model uses a **GradientBoostingClassifier** and is able to classify currencies with **~97% accuracy.**

---

## Exploratory Data Analysis (EDA)

The notebook includes visualizations to highlight the structure in the data:

### 1. Price Distribution by Currency
Shows distinct price ranges for EUR, AUD, and USD, with EUR tightly clustered, AUD appearing at higher price levels, and USD in the middle.

### 2. Time-Series Price Movements
All three currencies follow similar long-term trends but maintain consistent separation in their absolute price levels throughout the year.

### 4. Hour-of-Day and Global Session Patterns
Strong time-based signals:
- AUD appears almost exclusively during daytime hours and fully dominates the AU session
- EUR appears heavily during late-night + early-morning making it dominant in the EU session and a major presence in the US session
- USD concentrates around midnight, placing most of its activity into the US session and some into the EU session

### 5. Day-of-Week Patterns
- AUD is only quoted on weekdays
- USD is not quoted on Saturdays and very rarely on Mondays
- EUR is very rarely quoted on Saturdays, peaking Tuesday-Friday

---

## Modeling Pipeline

### Train/Validation/Test Split

the dataset is split strictly by timestamp using splicing:

- **70% Train**  
- **15% Validation**  
- **15% Test**

There is:

- No shuffling → preserves temporal order
- No stratification → keeps the market’s natural class imbalance

### Feature Engineering

The model uses the following features:

| Feature        | Description |
|----------------|-------------|
| `PRICE`        | Raw quote price |
| `PRICE_Z`      | Z-score normalized price using **training mean/std** |
| `HOUR`         | Hour extracted from timestamp (0–23) |
| `DAY_OF_WEEK`  | Integer day of week (0=Mon … 6=Sun) |
| `SESSION_US`   | 1 if hour ∈ [0, 8), else 0 |
| `SESSION_AU`   | 1 if hour ∈ [8, 16), else 0 |
| `SESSION_EU`   | 1 if hour ≥ 16, else 0 |

All engineered features can be computed from a **single incoming price + timestamp**, ensuring the model can run in real-time.

### Hyperparameter Tuning

All hyperparameter tuning is performed using the **validation set**, not the test set.  
This ensures the model does not indirectly learn from future data.

A grid search is run over combinations of:

- Number of trees: **150, 300, 500**
- Learning rates: **0.10, 0.05, 0.02**
- Tree depths: **2, 3, 4**

**Heatmaps** are generated to visualize the accuracy landscape for each depth level.

### Model Training

After selecting the optimal hyperparameters, the **training and validation sets are combined** to form a larger, more robust training dataset.  
This approach ensures that:

- The validation set is used strictly for selecting hyperparameters  
- The final model benefits from the maximum amount of available data  
- The test set remains completely untouched for unbiased evaluation  

The final Gradient Boosting model is trained using the best-performing configuration discovered during tuning:

- **150 trees**  
- **0.1 learning rate**  
- **max depth = 2**

### Model Evaluation

The **test set** is kept completely separate from training and validation to provide an unbiased measure of real-world performance.

The final model achieves **~97% accuracy** on the untouched test set.  
Performance varies by currency due to differences in price behavior and sample size:
```
Test Accuracy: 0.9675   (≈96.75%)

AUD: precision 1.00 | recall 1.00 | f1 = 1.00  
EUR: precision 0.94 | recall 1.00 | f1 = 0.97  
USD: precision 1.00 | recall 0.85 | f1 = 0.92  
```
- **AUD:** perfect classification (100% precision and recall)  
- **EUR:** strong performance with high recall and near-perfect f1-score  
- **USD:** slightly lower recall due to its smaller sample size and overlapping price region with EUR  

Evaluation includes:

- A **confusion matrix** showing few misclassifications of USD as EUR
- A **probability confidence distribution**, confirming the model generally predicts with high certainty  
- A **feature importance analysis**, highlighting which signals contribute most to classification decisions

Overall, the model generalizes well and maintains high accuracy across all classes while respecting the temporal structure of the data.

---

## Prediction Function

A production-style function `predict_currency(price, timestamp)` accepts:

- Any price value  
- Any timestamp (**past or future**)  

It constructs all required features and returns:

- Predicted currency  
- Prediction probability
---
