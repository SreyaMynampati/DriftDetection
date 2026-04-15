
---

# NOAA Storm Events — Real-Time Drift Detection and Prediction System

## Overview

This project implements a real-time machine learning system for analyzing NOAA Storm Events data. It combines classification, regression, concept drift detection, novelty detection, and adaptive retraining within a streaming simulation environment.

The system processes storm data in batches, continuously evaluates model performance, detects distribution shifts, and retrains models when necessary to maintain accuracy over time.

---

## Key Features

### Multi-Class Classification

* Model: Random Forest Classifier
* Target: EVENT_TYPE
* Handles class imbalance using class_weight = balanced
* Tracks accuracy, F1 (macro and weighted), and per-class metrics

---

### Multi-Target Regression

Separate Random Forest Regressors are trained for:

* Property Damage (in thousands of dollars)
* Storm Duration (in hours)
* Direct Injuries

Each regressor:

* Uses all other features as input
* Is evaluated using MAE, RMSE, and R²
* Is retrained when drift is detected

---

### Concept Drift Detection

Drift is detected using a hybrid approach combining:

* Kolmogorov–Smirnov (KS) Test
* Maximum Mean Discrepancy (MMD)
* Mean Change
* Performance Drop

Drift probability is computed as:

P(drift) = 0.25·KS + 0.25·MMD + 0.15·MeanChange + 0.35·PerformanceDrop

The system also includes predictive warnings based on trends in these signals.

---

### Novelty Detection

* Model: Isolation Forest
* Identifies unseen or anomalous storm patterns
* Outputs novelty rate and anomaly scores per batch

---

### Adaptive Retraining

When drift exceeds a threshold:

* Classifier and regressors are retrained
* Retraining cost and performance improvement are logged
* Walk-forward backtesting evaluates effectiveness

The system also compares:

* Performance with retraining
* Performance without retraining (frozen model)

---

### Drift Memory and Recurrence Detection

* Stores past drift signatures
* Uses cosine similarity to identify recurring patterns
* Distinguishes between known and novel drift

---

### Explainability

* Per-feature drift attribution using KS statistics
* Population Stability Index (PSI) for feature shifts
* Random Forest feature importance

---

### Interactive Dashboard

Built using Streamlit, providing:

* Real-time monitoring
* Drift timeline visualization
* Performance tracking
* Per-class metrics
* Regression analysis

---

## Dataset

The system uses NOAA Storm Events CSV files.

Expected directory structure:

```
datasets/
├── StormEvents_details-ftp_v1.0_*.csv
```

Features used:

* Latitude, Longitude
* Magnitude
* Duration
* Injuries and Deaths
* Property and Crop Damage

---

## Tech Stack

* Python
* Streamlit
* Scikit-learn
* NumPy
* Pandas
* SciPy
* Plotly

---

## Installation

Clone the repository:

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Application

```
streamlit run storm.py
```

---

## System Workflow

1. Load and preprocess NOAA storm data
2. Simulate streaming using batch generation
3. Train initial classifier and regressors
4. For each batch:

   * Perform predictions
   * Evaluate performance
   * Detect drift
   * Detect novelty
   * Retrain models if needed
5. Visualize results in real time

---

## Metrics

### Classification

* Accuracy
* F1 Score (macro and weighted)
* Precision and Recall per class

### Regression

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

### Drift Detection

* KS Statistic
* MMD Score
* Drift Probability
* Warning Score

---

## Use Cases

* Monitoring non-stationary data streams
* Climate and environmental analytics
* Real-time anomaly detection
* Adaptive machine learning systems
* Concept drift research

---
