# Bank Marketing Campaign — Deposit Prediction

A capstone project for CSE 475 (Foundations of Machine Learning) that builds and evaluates a binary classifier to predict whether a bank customer will make a term deposit based on data from a direct marketing campaign.

**Dataset:** [Bank Marketing Dataset — Kaggle / UCI ML Repository](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)

---

## Problem Statement

Given information about a bank's direct marketing campaign (phone calls), predict whether a client will subscribe to a term deposit (`yes` / `no`). This is a **binary classification** problem where the target variable is `deposit`.

---

## Dataset

| Feature | Description |
|---|---|
| `age` | Client age |
| `job` | Type of job |
| `marital` | Marital status |
| `education` | Education level |
| `default` | Has credit in default? |
| `balance` | Average yearly account balance |
| `housing` | Has housing loan? |
| `loan` | Has personal loan? |
| `contact` | Contact communication type |
| `day` / `month` | Last contact date |
| `duration` | Last contact duration (seconds) |
| `campaign` | Number of contacts during this campaign |
| `pdays` | Days since last contact from previous campaign |
| `previous` | Number of contacts before this campaign |
| `poutcome` | Outcome of previous marketing campaign |
| `deposit` | **Target** — did the client subscribe? |

**Size:** 11,162 rows × 17 columns (pre-encoding)

---

## Project Structure

The project is organized across three milestones:

### Milestone 1 — EDA & Preprocessing
- Loaded and explored the dataset
- Checked for null values and identified data quality issues
- Visualized age distribution, age vs. balance, and balance by job type
- Noted that some jobs (management, entrepreneur, self-employed) have negative balances but kept them as they are not severe outliers
- Applied **one-hot encoding** to all categorical columns, expanding to 56 features

### Milestone 2 — Model Training & Evaluation
- Chose **logistic regression** as the model, implemented from scratch in **PyTorch**
- Target: `deposit_yes` (binary — 1 if customer made a deposit, 0 otherwise)
- Train/test split: **80/20** with stratification
- Used **K-Fold cross-validation** (5 folds) with an SVM to tune hyperparameters (learning rate, weight decay)
- Evaluated with **precision, recall, F1 score, confusion matrix, and ROC curve**
- Achieved ~66% accuracy (1,473/2,233 correct predictions on test set)

### Milestone 3 — Comparison, Deployment & Ethics
- Compared logistic regression model against **scikit-learn Random Forest** (100 estimators)
- Used **SHAP** to explain feature importance for both models
- Key finding: `duration`, `balance`, and `pdays` are the top predictors in both models; Random Forest better captures non-linear relationships

---

## Model Architecture

```python
class LogisticRegModel(nn.Module):
    def __init__(self, input):
        super().__init__()
        self.linear = nn.Linear(input, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))
```

**Training config:**
- Loss: Binary Cross Entropy (`BCELoss`)
- Optimizer: SGD (lr=0.01)
- Epochs: 1,000
- Batch size: 32

---

## Results

| Metric | Value |
|---|---|
| Accuracy | ~66% |
| Key features (SHAP) | `duration`, `balance`, `pdays` |

Both the custom logistic regression and the Random Forest agreed on the most important features. The Random Forest performed better on non-linear feature interactions, suggesting feature engineering or a more complex model could improve results.

---

## Deployment Plan

- **Model:** Logistic Regression (PyTorch)
- **API:** FastAPI served via Uvicorn
- **Containerization:** Docker (scalable via Kubernetes for load balancing)
- **Data pipeline:** Clean → transform → store in Snowflake
- **Monitoring:** System + model monitoring with logs saved to AWS
- **Dashboard:** UI for monitoring systems and documentation

---

## Ethical Considerations

- **Bias:** Model may unfairly predict low-balance customers as unlikely to deposit, which could affect decisions about who to target in campaigns
- **Transparency:** Clear communication to stakeholders about how predictions are made
- **Privacy:** No personal data stored in plain text; encryption required for any stored PII
- **Accountability:** Decision logs retained; users should be able to challenge model predictions
- **Compliance:** Aligned with data protection regulations; minimal data collection principle applied

---

## Requirements

```
torch
pandas
numpy
matplotlib
seaborn
scikit-learn
shap
```

---

## Data Source

UCI Machine Learning Repository — Bank Marketing Dataset  
https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset
