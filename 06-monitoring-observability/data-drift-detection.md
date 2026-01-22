# Data Drift Detection

## Overview

Data drift occurs when input data distribution changes, leading to model performance degradation. Detecting drift early enables proactive model updates.

---

## 🎯 Types of Drift

### 1. Concept Drift
- Target relationship changes
- Model becomes less accurate
- Requires retraining

### 2. Data Drift
- Input distribution changes
- Feature values shift
- May require retraining

### 3. Covariate Shift
- Input distribution changes
- Target relationship unchanged
- May not require retraining

---

## 🛠️ Detection Methods

### Statistical Tests
- Kolmogorov-Smirnov test
- Chi-square test
- Mann-Whitney U test

### Distance Metrics
- Wasserstein distance
- Kullback-Leibler divergence
- Jensen-Shannon divergence

### Machine Learning
- Drift detection models
- Anomaly detection
- Change point detection

---

## ✅ Best Practices

1. **Monitor continuously** - detect drift early
2. **Use multiple methods** - combine approaches
3. **Set thresholds** - meaningful drift levels
4. **Automate retraining** - trigger on drift
5. **Document drift** - track patterns

---

## 🔗 Related Topics

- [Model Monitoring](./model-monitoring.md)
- [Performance Metrics](./performance-metrics.md)
