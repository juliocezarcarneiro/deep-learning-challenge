# Neural Network Model Report: Alphabet Soup Analysis

# Overview of the Analysis
The goal of this analysis is to develop and evaluate a deep learning model to predict the success of charitable applications submitted to Alphabet Soup. Using historical data, the task is framed as a binary classification problem where the model predicts whether an application will be successful (IS_SUCCESSFUL = 1) or not (IS_SUCCESSFUL = 0). The model’s performance is compared against a benchmark accuracy target of 75%.

# Results 

## Data Preprocessing
* Target Variable:

    * IS_SUCCESSFUL

* Feature Variables:

    * All other columns in the preprocessed dataset after converting categorical features with pd.get_dummies, except those removed for being identifiers or not useful for prediction.

* Variables Removed:

    * Columns such as EIN and NAME were removed since they are identifiers and provide no predictive power.

    * Rare categories in CLASSIFICATION and APPLICATION_TYPE were grouped into "Other" to reduce noise and dimensionality.

## Compiling, Training, and Evaluating the Neural Network Model
* Model Architecture:

    * Input Layer: Based on the number of features after scaling.

    * First Hidden Layer: 128 neurons, ReLU activation

    * Second Hidden Layer: 64 neurons, ReLU activation

    * Output Layer: 1 neuron, Sigmoid activation (binary classification)

* Compilation Settings:

    * Loss Function: Binary Crossentropy

    * Optimizer: Adam

    * Metrics: Accuracy

* Training Results:

    * Final Accuracy: 0.7305

    * Loss: 0.5713

📉 Model Performance (Neural Network):

* The model achieved 73.05% accuracy, which did not meet the 75% target.

* The loss function indicates moderate error, suggesting some room for optimization.

* Steps Taken to Improve Performance:

    * Tried different numbers of neurons in the hidden layers.

    * Normalized input features using StandardScaler.

    * Used categorical grouping and encoding to simplify features.

    * Increased epochs and adjusted batch size for better training convergence.

📊 XGBoost Benchmark Comparison
* Optimized with Optuna Hyperparameter Tuning:

    * Parameters such as n_estimators, learning_rate, max_depth, subsample, and colsample_bytree were fine-tuned over 50 trials.

* Final Evaluation (on full dataset):

    * Accuracy: 0.7502

    * ROC AUC: 0.8155

# Summary
The deep learning model achieved a final accuracy of 73.05%, falling slightly short of the 75% performance benchmark. Despite standard preprocessing, feature engineering, and tuning hidden layers, the neural network did not outperform the XGBoost model, which reached 75.02% accuracy and a strong ROC AUC of 0.8155.

## Recommendation:
For this binary classification task, XGBoost is a better choice due to its:

* Higher predictive accuracy

* Better handling of feature importance