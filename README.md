# 🤖 Alphabet Soup: Neural Network vs. XGBoost for Donation Success Prediction

## 📌 Project Overview

The goal of this project is to build a deep learning model to predict whether charitable donation applications submitted to Alphabet Soup will be successful. Using historical application data, we frame this as a **binary classification** problem (`IS_SUCCESSFUL = 1` or `0`) and benchmark our model against a **75% accuracy target**.

Additionally, we compare the neural network's performance to a tuned **XGBoost model**, to determine which approach is more effective for this problem.

---

## 🧠 Neural Network Model Report

### 🔄 Data Preprocessing

- **Target Variable**:  
  - `IS_SUCCESSFUL`

- **Feature Variables**:  
  - All other columns in the cleaned dataset after encoding categorical features with `pd.get_dummies`.

- **Removed Variables**:
  - `EIN`, `NAME` — dropped as identifiers with no predictive value.
  - Rare values in `CLASSIFICATION` and `APPLICATION_TYPE` were grouped under `"Other"` to reduce dimensionality and noise.

---

### 🛠️ Model Architecture & Training

- **Input Layer**:
  - Based on the number of features after scaling.

- **Hidden Layers**:
  - Layer 1: 128 neurons, ReLU
  - Layer 2: 64 neurons, ReLU

- **Output Layer**:
  - 1 neuron, Sigmoid activation (for binary classification)

- **Compilation**:
  - Loss Function: `binary_crossentropy`
  - Optimizer: `adam`
  - Metric: `accuracy`

- **Preprocessing**:
  - Numeric features scaled with `StandardScaler`
  - Categorical features encoded with `pd.get_dummies`

---

### 📉 Neural Network Performance

- **Accuracy**: `73.05%`
- **Loss**: `0.5713`

#### ⚙️ Optimization Steps

- Experimented with different:
  - Hidden layer sizes and neuron counts
  - Epochs and batch sizes
  - Feature groupings for rare classes
- Scaled inputs for better training stability

⛔ **Conclusion**: Despite tuning, the neural network fell short of the 75% accuracy benchmark.

---

## 🔁 XGBoost Benchmark (with Optuna Tuning)

- **Optimization**:
  - Used `Optuna` with 50 trials to fine-tune:
    - `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`

- **Final Evaluation**:
  - Accuracy: `75.02%`
  - ROC AUC: `0.8155`

- **Classification Report**:
  - Precision, recall, and F1-scores exceeded those of the neural network.

---

## 🧾 Summary & Recommendation

The neural network performed reasonably well, achieving **73.05% accuracy**, but did not meet the benchmark. The **XGBoost model**, optimized with Optuna, reached **75.02% accuracy** and demonstrated better performance overall.

### ✅ Recommendation:
For this binary classification task:
- Use **XGBoost** due to:
  - Higher accuracy
  - Better handling of categorical and imbalanced features
  - Easier interpretability and feature importance extraction

---

## 💻 Technologies Used

- Python 3.11
- TensorFlow / Keras
- Scikit-learn
- XGBoost
- Optuna (for hyperparameter tuning)
- Pandas, NumPy, Matplotlib

---

## 🚀 How to Run the Project

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/alphabet-soup-analysis.git
   cd alphabet-soup-analysis
