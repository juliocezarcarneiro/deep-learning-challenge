# Alphabet Soup Charity Funding Predictor

### Overview
This project aims to help Alphabet Soup, a nonprofit foundation, identify which funding applicants have the highest likelihood of success. Using machine learning and neural networks, we've developed a binary classifier that predicts whether an organization will use funding effectively based on various metadata features.

### Dataset Information
The dataset contains metadata for over 34,000 organizations that have received funding from Alphabet Soup, including:

* Identification columns: EIN, NAME

* Application details: APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE

* Organization info: ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS

* Funding details: ASK_AMT (funding amount requested)

* Target variable: IS_SUCCESSFUL (whether money was used effectively)

### Implementation Steps
1. Data Preprocessing
    * Target variable: IS_SUCCESSFUL (binary classification)

    * Features: All relevant columns except EIN and NAME

    * Preprocessing steps:

        * Dropped non-beneficial columns (EIN, NAME)

        * Binned rare categorical values

        * Encoded categorical variables using one-hot encoding

        * Scaled features using StandardScaler

        * Split data into training and testing sets

2. Neural Network Model
Initial Model Architecture:

Input layer matching number of features

Two hidden layers with ReLU activation

Output layer with sigmoid activation

Compiled with binary_crossentropy loss and adam optimizer

3. Model Optimization
Attempted various optimization techniques including:

Adjusting binning thresholds for categorical variables

Adding/removing hidden layers

Experimenting with different activation functions

Varying the number of neurons per layer

Adjusting training epochs

### Results
Dawta Preprocessing
Target Variable: IS_SUCCESSFUL

Features: APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT

Removed Columns: EIN, NAME (identification only)

Model Performance
Initial Model Accuracy: [X]%

Optimized Model Accuracy: [Y]%

Optimization Techniques Applied:

Technique 1 (e.g., added hidden layer)

Technique 2 (e.g., adjusted binning)

Technique 3 (e.g., changed activation functions)

Usage
To use the trained model:

Load the HDF5 file using tensorflow.keras.models.load_model()

Preprocess new data using the same steps as in the notebooks

Make predictions using model.predict()

Dependencies
Python 3.7+

TensorFlow 2.0+

pandas

scikit-learn

Jupyter Notebook

Recommendations
While the neural network achieved [X]% accuracy, alternative approaches like Random Forest or Gradient Boosting might provide better interpretability with comparable performance. Future work could explore:

Feature engineering to create more predictive features

Hyperparameter tuning with Keras Tuner

Ensemble methods combining multiple models

License
This project is licensed under the MIT License.

Credits
Toronto University and edX