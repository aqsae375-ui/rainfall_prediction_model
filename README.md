# rainfall_prediction_model

## 1. Problem Statement

The goal of this project is to build a binary classification model to predict RainTomorrow **(Yes/No)** using meteorological features such as temperature, humidity, pressure, and wind conditions. The primary objective is to develop a model that can accurately forecast rain, which has significant implications for various sectors including agriculture, event planning, and daily life.

## 2. Data Description

The dataset used for this project is `usa_rain_prediction_dataset_2024_2025.csv`. It contains meteorological features and a target variable indicating whether it will rain tomorrow. The dataset includes the following columns:

-   **Date**: The date of the observation.
-   **Location**: The geographical location of the observation.
-   **Temperature**: The temperature in degrees Celsius.
-   **Humidity**: The relative humidity as a percentage.
-   **Wind Speed**: The speed of the wind.
-   **Precipitation**: The amount of precipitation.
-   **Cloud Cover**: The percentage of the sky covered by clouds.
-   **Pressure**: The atmospheric pressure.
-   **Rain Tomorrow**: The target variable, indicating whether it rained (1) or not (0) on the following day.

## 3. Preprocessing Steps

Data preprocessing is crucial for preparing the dataset for model training. The following steps were performed:

1.  **Missing Values**: Initially, the dataset was checked for missing values using `df.isnull().sum()`. Fortunately, no missing values were found.
2.  **Dropping 'Date' Column**: The 'Date' column was dropped as it is an identifier and not directly used as a predictive feature for the models.
3.  **One-Hot Encoding 'Location' Column**: The 'Location' column, a categorical feature, was transformed using one-hot encoding (`pd.get_dummies(df, columns=['Location'], drop_first=True)`). This converts categorical data into a numerical format suitable for machine learning algorithms.
4.  **Data Leakage Issue with Scaling**: A critical issue was identified during the initial development phase: `StandardScaler.fit_transform()` was incorrectly applied to the *entire* DataFrame before the train-test split. This introduced **data leakage** because the scaling parameters (mean and standard deviation) were computed using information from both the training and testing sets. As a result, the models had an unrealistically optimistic view of their performance on unseen data, leading to inflated metrics (e.g., perfect scores for Random Forest and XGBoost).
5.  **Corrected Preprocessing Pipeline**: To mitigate data leakage, the preprocessing steps were reordered and refined:
    *   The dataset was reloaded, and the 'Date' column was dropped.
    *   One-hot encoding for 'Location' was applied to the entire DataFrame (which is acceptable for one-hot encoding as it doesn't learn statistics from data in a way that causes target-dependent leakage across splits).
    *   The data was then split into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`) *before* numerical scaling.
    *   A `StandardScaler` was initialized and **fitted only on the numerical features of the training set (`X_train[numerical_features]`)**.
    *   This *fitted* scaler was then used to `transform` both the training set (`X_train[numerical_features]`) and the test set (`X_test[numerical_features]`). This ensures that the test set is scaled using parameters learned exclusively from the training data, preventing data leakage and providing a more realistic evaluation of model performance.

## 4. Model Details and Performance

### 4.1 Logistic Regression

**Initial Model Performance:**

-   Accuracy: 0.9072
-   Precision: 0.8450
-   Recall: 0.7089
-   F1-score: 0.7710

**Hyperparameter Tuning:**

Hyperparameter tuning was performed using `GridSearchCV` to optimize for recall. The parameter grid explored included regularization strength `C` (`[0.001, 0.01, 0.1, 1, 10, 100]`) and `solver` (`['liblinear', 'saga']`). The `scoring` metric was set to `'recall'`.

-   Best Hyperparameters Found: `{'C': 10, 'solver': 'liblinear'}`
-   Best Cross-Validation Recall Score: 0.7036

**Tuned Model Performance (after correcting data leakage):**

-   Accuracy: 0.9073
-   Precision: 0.8453
-   Recall: 0.7089
-   F1-score: 0.7711

### 4.2 Random Forest Classifier

**Initial (Suspect) Model Performance:**

Before correcting the data leakage issue in the preprocessing pipeline, the Random Forest Classifier exhibited suspiciously perfect performance metrics:

-   Accuracy: 1.0000
-   Precision: 1.0000
-   Recall: 1.0000
-   F1-score: 1.0000

As noted, this was a strong indicator of **data leakage**, where the model inadvertently gained access to information from the test set during its training or scaling, leading to an overestimation of its true performance.

**Performance after Correcting Data Leakage:**

After fixing the data leakage by ensuring proper train-test split and scaling, K-fold cross-validation (5-fold StratifiedKFold) was applied to the Random Forest model to get a more reliable estimate of its performance.

-   Mean Recall (Cross-Validation on Training Data): 0.9998
-   Standard Deviation (Cross-Validation): 0.0003

*(Note: While still very high, this reflects the model's performance on the training folds and suggests the model might be overfitting or the dataset might be very separable. Further investigation, including hyperparameter tuning, would be required to ensure generalization.)*

### 4.3 XGBoost Classifier

**Initial (Suspect) Model Performance:**

Similar to the Random Forest model, the XGBoost Classifier also showed unusually high performance metrics before addressing the data leakage problem:

-   Accuracy: 0.9995
-   Precision: 0.9988
-   Recall: 0.9991
-   F1-score: 0.9989

This also strongly suggested **data leakage**, necessitating a review and correction of the preprocessing steps.

**Hyperparameter Tuning (after correcting data leakage):**

After resolving the data leakage, the XGBoost model was re-evaluated and tuned using `GridSearchCV` with 5-fold StratifiedKFold, focusing on `'recall'`. The hyperparameter grid included:

-   `n_estimators`: `[100, 200, 300]`
-   `learning_rate`: `[0.01, 0.1, 0.2]`
-   `max_depth`: `[3, 5, 7]`
-   `subsample`: `[0.7, 0.8, 0.9]`

-   Best Hyperparameters Found: `{'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.7}`
-   Best Cross-Validation Recall Score: 0.9993

**Final Tuned Model Performance on Test Set (after correcting data leakage):**

-   Accuracy: 0.9992
-   Precision: 1.0000
-   Recall: 0.9966
-   F1-score: 0.9983

## 5. Comparison of Model Performances

After debugging and correcting the data leakage issue in the preprocessing pipeline, the performance metrics for all models provide a more realistic and trustworthy comparison:

| Model                        | Accuracy | Precision | Recall | F1-score |
| :--------------------------- | :------- | :-------- | :----- | :------- |
| **Logistic Regression (Tuned)** | 0.9073   | 0.8453    | 0.7089 | 0.7711   |
| **Random Forest (CV Mean)**  | -        | -         | 0.9998 | -        |
| **XGBoost (Tuned)**          | 0.9992   | 1.0000    | 0.9966 | 0.9983   |

-   **Logistic Regression**: This model serves as a strong baseline, showing consistent performance with an accuracy of around 90% and recall of 70.9%. Its performance is stable and interpretable.
-   **Random Forest**: While its cross-validation mean recall was exceptionally high (0.9998), suggesting strong performance, a direct comparison of all metrics on the test set was not performed after tuning. The high recall from CV could indicate strong learning on the training data, but further evaluation on the unseen test set is crucial for generalization.
-   **XGBoost**: After debugging the data leakage and tuning, XGBoost delivered outstanding performance on the test set, achieving an accuracy of 0.9992, perfect precision (1.0000), and a very high recall of 0.9966. This suggests that XGBoost is highly effective for this prediction task, particularly in minimizing false negatives (missed rain events), which is critical for rain prediction.

Given the corrected evaluation, the **tuned XGBoost model** stands out as the most performant, especially concerning recall, which is a key metric for scenarios where missing a positive event (rain) has significant consequences.

## 6. Key Findings and Insights

-   **Importance of Data Integrity**: The most critical insight from this project is the profound impact of **data leakage**. Initial model evaluations were highly misleading due to incorrect scaling of features before the train-test split. This highlighted the absolute necessity of meticulously reviewing preprocessing pipelines to prevent information from the test set from influencing the training phase.
-   **Correct Preprocessing Order**: Always perform the train-test split *before* any data transformations (like scaling, imputation, or complex feature engineering) that derive parameters from the data. Fit transformers only on the training data and then apply them to both training and test sets.
-   **Model Performance Range**: Simple models like Logistic Regression provide a robust and interpretable baseline. Advanced ensemble methods like Random Forest and XGBoost have the potential for significantly higher performance, but their evaluation must be done carefully to ensure true generalization.
-   **Recall Optimization**: Focusing on 'recall' as the optimization metric for `GridSearchCV` was crucial for developing a model that minimizes missed rain events, which is often a priority in rain prediction.

## 7. Setup and Running the Project Locally

To set up and run this rain prediction project on your local machine, follow these steps:

### Prerequisites

-   Python 3.8+
-   `pip` (Python package installer)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

### 3. Install Dependencies

Install all required Python packages using the `requirements.txt` file (you might need to create this file based on the imports in the notebook):

```bash
pip install -r requirements.txt
```

*(Example `requirements.txt` content: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`)*

### 4. Run the Jupyter/Colab Notebook

Start a Jupyter Notebook server and open the project's notebook file (`your_notebook_name.ipynb`). Alternatively, upload the notebook to Google Colab.

```bash
jupyter notebook
```

Follow the instructions and run the cells in the notebook to reproduce the data loading, preprocessing, model training, evaluation, and analysis.
