# Project: Disease Analysis & Prediction System

This project is a comprehensive Python-based analytical tool designed to process medical data, perform exploratory data analysis, and build a predictive model for Diseases - for example Coronary Artery Disease (CAD). 
It features advanced data visualization, Dimensionality Reduction (PCA), and a Random Forest Classifier to estimate patient risk.

## Features

* **Target Transformation**: Automatically converts multi-level diagnosis data into a binary classification (Healthy vs. Sick).
* **Dimensionality Reduction**: Implements PCA (Principal Component Analysis) to visualize complex all attribute medical data in a 2D space.
* **Detailed Exploratory Data Analysis(EDA)**: Generates side-by-side histograms for all medical attributes to compare distributions between sick and healthy populations.
* **Predictive Modeling**: Utilizes a Random Forest Classifier to predict the probability of the disease for new patients.
* **Performance Evaluation**: Calculates AUC (Area Under Curve) and plots ROC curves to measure model reliability.
* **Feature Importance**: Identifies and ranks the clinical attributes that contribute most to a successful diagnosis.

## Analytical Methods

| Method | Action | Description |
| :--- | :--- | :--- |
| `plot_pca()` | **Visualization** | Standardizes data and reduces dimensions to 2D for cluster analysis. |
| `plot_histograms()` | **EDA** | Creates subplots showing the distribution of attributes using `dodge` or `stack` layouts. |
| `_train_rf_model()` | **Training** | Trains the Random Forest model on 70% of the dataset with a fixed `random_state`. |
| `plot_roc()` | **Evaluation** | Generates the ROC curve and computes the `AUC` score for the test set. |
| `get_top_features()` | **Inference** | Extracts the relative contribution of each medical feature to the prediction. |

## Technologies Used

* **Programming Language**: Python (3.x)
* **Data Manipulation**: `pandas`, `numpy`
* **Machine Learning**: `scikit-learn` (`RandomForestClassifier`, `PCA`, `StandardScaler`)
* **Visualization**: `matplotlib`, `seaborn`

## Setup and Usage

### 1. Prerequisites
* Install Python 3.x.
* Install required libraries:
```bash
pip install pandas matplotlib seaborn scikit-learn numpy
