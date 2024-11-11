# Parkinson's Disease Detection using Machine Learning

This repository contains a machine learning project for detecting Parkinson's Disease using data from a Kaggle dataset. The project is implemented in **Google Colab** and aims to predict the presence of Parkinson's Disease based on various features related to speech and medical tests.

## Dataset

- **Source**: [Parkinson's Disease Data on Kaggle](https://www.kaggle.com/datasets/ashishpatel26/parkinsons-disease-detection)
- **Description**: The dataset consists of various medical and speech-related features such as jitter, shimmer, and other voice-based characteristics that can be used to predict the presence of Parkinson’s Disease.
- **Features**:
  - Speech-related features like pitch, jitter, and shimmer.
  - Medical features like age, gender, and others.

## Tools & Libraries Used

- **Python**
- **Google Colab** for implementation and running the model.
- **Machine Learning Libraries**:
  - `scikit-learn` for building and evaluating the model.
  - `pandas` and `numpy` for data manipulation and analysis.
  - `matplotlib` and `seaborn` for visualization.

## Project Details

### 1. **Data Preprocessing**:
   - The data was cleaned and scaled to ensure that all features are within a similar range.
   - The data was split into training and testing sets for model evaluation.

### 2. **Model Development**:
   - **Support Vector Machine (SVM)** was used as the primary model to predict Parkinson’s Disease.
   - Hyperparameter tuning was performed to optimize the SVM model for better accuracy.

### 3. **Model Evaluation**:
   - The model was evaluated using metrics such as accuracy, precision, recall, and F1-score.
   - A confusion matrix and classification report were generated to provide insights into the model's performance.

### 4. **Conclusion**:
   - The SVM model achieved satisfactory performance in detecting Parkinson's Disease, with potential for further improvements.
   - Future work could explore additional models or feature engineering for even better predictions.
