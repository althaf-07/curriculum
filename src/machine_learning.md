# Full Classical Machine Learning Curriculum (without Mathematics and Deep Learning)

---

## 1. Introduction to Machine Learning

---

### 1.1 What is Machine Learning?

---

### 1.2 Machine Learning Paradigms (also known as Types of Machine Learning)

#### 1.2.1 Supervised Learning

- Tasks:
  - Classification
  - Regression

#### 1.2.2 Unsupervised Learning

- Tasks:
  - Clustering
  - Dimensionality Reduction
  - Association Rule Learning

#### 1.2.3 Semi-supervised Learning

- Techniques:
  - Self-Training (Pseudo-Labeling)
  - Consistency Regularization
  - Graph-Based Semi-Supervised Learning
  - Low-Density Separation (Cluster Assumption)

#### 1.2.4 Reinforcement Learning

- Key Strategies:
  - Model-Based RL vs. Model-Free RL
  - Value-Based vs. Policy-Based vs. Actor-Critic RL
  - On-Policy vs. Off-Policy RL
  - Exploration vs. Exploitation Strategies in RL
  - Multi-Agent Reinforcement Learning (MARL)

---

### 1.3 Machine Learning Libraries

### 1.3.1 Numerical and Statistical Computations 

- Numpy
- Scipy
- Statsmodels

### 1.3.2 Data Manipulation

- Pandas
- Polars

### 1.3.3 Data Visualization

- Matplotlib
- Seaborn
- Plotly

### 1.3.4 Machine Learning Algorithms and Tools

- Scikit-learn (also known as Sklearn)
- Category-encoders
- Imbalanced-learn (also known as Imblearn)
- Gradient Boosting Frameworks
  - XGBoost
  - LightGBM
  - CatBoost
- Optuna

### 1.3.5 Miscellaneous

- Joblib
- MLflow

---

### 1.4 Bias, Variance, and Generalization

#### 1.4.1 Bias and Variance

- Bias
- Variance
- Bias-Variance Tradeoff

#### 1.4.2 Generalization

- Underfitting
- Overfitting

---

## 2 Data: The Gold and Oil of the 21st Century

### 2.1 Types of Data

### 2.1.1 Data Structure

- Structured
- Unstructured
- Semi-structured Data

### 2.1.2 Data Formats

- Tabular
- Text
- Image
- Audio
- Video
- Time-Series Data
- Sensor Data

### 2.1.3 Types of Columns

- Numerical
- Categorical
  - Numerical Categorical
  - Binary Categorical
  - Multi Categorical
- Time Series Columns

---

### 2.2 Understanding the Dataset

#### 2.2.1 Dataset Overview

- Number of Rows
- Number of Columns
- Names of Columns
- Data Types for Each Column
- Numerical Features:
  - Count of all the values in a column
  - Minimum (Min)
  - Mean (Average)
  - 25th Percentile (also known as 1st Quartile)
  - Median (also known as 50th Percentile and 2nd Quartile)
  - 75th Percentile (also known as 3rd Quartile)
  - Standard Deviation (std.)
  - Skewness
  - Kurtosis
  - Maximum (Max)
- Categorical Features:
  - Count of all the values in a column
  - Number of Unique Values in a column
  - Most Frequent Value (also known as Mode)
  - Frequency of the Most Frequent Value
  - Least Frequent Value
  - Frequency of the Least Frequent Value

#### 2.2.2 Dataset Overview prior to Data Cleaning

- Features with Missing Values
- Rows with Duplicate Data Points
- Numerical Features with Outliers
- Imbalanceness of Target Variable (If Classification problem)

## 4. Handling Missing Values

### 4.1 Important Questions

- If values in target column are missing, remove that entire row. Don't try to impute that.
- If a feature column has more than 50% of missing values, remove that entire feature.
- If a row has more than 50% of missing values, remove that entire row
- Does target variable has missing values? If 'Yes' then remove that data points.
- Names of Numerical columns that has missing values
- Names of Categorical columns that has missing values
- Does any column has more than ? If 'Yes' then remove that column.
- Does any row has more than 50% of the number of columns? If 'Yes' then remove that data point.

### 4.2 Types of Missing Values

- MAR
- MCAR
- MNAR
- Learn how to handle each of them separately

- Removing
```python
      - Remove Rows if Target Variable 
      - Columns
```
---

    - When to remove or impute missing values
    
    - Imputation
      - Numerical
        - Univariate Imputation
          - Mean
          - Median
        - Multivariate Imputation
          - KNN Imputer
          - Iterative Imputer
          - Imputation using Regression Model
      - Categorical
        - Univariate
          - Mode
        - Multivariate
          - Imputation using Classification Model
      - Time Series Data
        - Forward Fill
        - Backward Fill
        - Interpolation
    - Indicator Variable for Missingness
      - When should we use indicator?
        - When the missing value itself has an importance in prediction. For example, if some variables in your problem is optional for the end user to provide, then its better to indicate them instead of removing or imputing. But, if you want to limit your end users to provide answers for all the variables, then its better to handle them instead of indication.

---

- Data Splitting Techniques
  - Training, Validation, and Testing Sets
  - Cross-Validation (k-fold, LOOCV, Stratified Sampling, Nested CV)
  - Temporal Splitting for Time-Series Data

---

- Data Cleaning
  - Standardization of strings in the dataset to `snake_case` (You can also try other cases like `PascalCase` or `camelCase`. But using `snake_case` is recommended, since Python prefers it. Also it is easier to read and looks clean. Whatever you choose, ensure you and your teammates stick to it in order to avoid future confusions).
    - Column Names
    - Categorical Values

    ---

  - Handling Duplicates

    ---

  - Handling Outliers
    - Difference between Data Error (e.g. Age in negative or greater than 120) and Actual Outliers (e.g. A student having very high marks on a hard test, while all of the other students scored low).
    - Identifying Outliers
    - Why should you keep or remove Outliers
    - Removing Outliers
    - Capping (also know as Winsorization)
      - Z-Score Method
      - IQR (Interquartile Range)
    - Clipping
      - Custom Value Clipping (Based on your domain knowledge)
      - Percentile based Clipping
    - Feature Transformation or Scaling to reduce impact of Outliers

    ---

  - Handling Imbalance Dataset
    - SMOTE (Synthetic Minority Oversampling Technique)
    - Oversampling and Undersampling
    - Class Weighting
    ---

- Data Scaling and Normalization
  - Scaling vs. Normalization
  - StandardScaler
  - MinMaxScaler
  - RobustScaler

- Categorical Variable Encoding
  - One-hot Encoding
  - Label Encoding
  - Target Encoding
  - Frequency Encoding

- Curse of Dimensionality
  - Impact on High-dimensional Datasets
  - Mitigation Methods (e.g., Feature Selection, Dimensionality Reduction)

- Data Visualization
  - Univariate Analysis
    - Numerical
      - Histogram
      - KDE Plot
      - Box Plot
      - Point Plot
    - Categorical
      - Bar Plot
        - Count Plot (a special type of Bar Plot)
      - Pie Chart
  - Bivariate Analysis
    - Numerical - Numerical
      - Scatter Plot
      - Pair Plot
      - Correlation Matrix Heatmap
    - Categorical - Categorical
      - Contingency Table Heatmap (also know as Crosstab Heatmap)
      - Cluster Heatmap
    - Numerical - Categorical
      - Histogram
      - KDE Plot
      - Box Plot
      - Line Plot
      - Bar Plot
  - Visualizations to Understand Distributions, Correlations, and Outliers

- Feature Engineering:
  - Feature Extraction
  - Feature Scaling
    - Standardization and Normalization
    - Scaling vs. Normalization
    - Techniques: StandardScaler, MinMaxScaler, RobustScaler
  - Feature Construction
  - Feature Transformation
    - Log Transformation
    - Square Transformation
    - Square Root Transformation
    - Reciprocal Transformation
    - Power Transformation
      - Box-Cox
      - Yeo-Johnson
  - Feature Selection
  - Feature Transformation
    - Feature Interaction (Polynomial Features, Crossed Features)
    - Dimensionality Reduction (PCA, t-SNE, UMAP)

  - Feature Importance

## Machine Learning Algorithms

### 4.3 Random Forest
- Number of Trees (n_estimators)
- Max Depth
- Max Features
- Bootstrap

### 4.4 Gradient Boosting (e.g., XGBoost, LightGBM, CatBoost)
- Learning Rate
- Number of Trees
- Max Depth
- Subsample
- Colsample_bytree

### Supervised Learning Algorithms

#### Linear Regression

- Assumptions
  - Linearity
  - Independence of Errors (No Autocorrelation)
  - Homoscedasticity
  - Normality of Residuals
  - No Multicollinearity
  - No Endogeneity (No Omitted Variable Bias)
- Common Hyperparameters
  - loss
  - alpha
  - penalty
  - l1_ratio
  - solver
  - max_iter
  - tol
- Variations
  - Ordinary Least Squares (OLS)
  - Stochastic Gradient Descent Regressor (SGD Regressor)
  - Ridge Regression
  - Lasso Regression
  - Polynomial Regression
  - Elastic Net Regression

#### Logistic Regression

- Assumptions
  - Binary Dependent Variable
  - Independence of Observations
  - Linearity of the Logit for Continuous Predictors
  - No (or Little) Multicollinearity
- Common Hyperparameters
  - C (Regularization Strength)
  - penalty
  - l1_ratio
  - solver
  - max_iter
  - tol
- Variations
  - Binary Logistic Regression
  - Multinomial Logistic Regression (Softmax Regression)
  - Ordinal Logistic Regression

#### Support Vector Machine (SVM)

- Assumptions
  - Data Separability
  - Balanced Classes
  - Use of Kernels
- Common Hyperparameters
  - C (Regularization Strength)
  - kernel
  - gamma
  - degree
  - coef0
  - max_iter
  - tol
- Terms
  - Kernel Trick
  - Kernel Methods

#### K-Nearest Neighbors (KNN)

- Assumptions
  - Similarity Assumption
  - Appropriate Distance Metric
  - Choice of K
- Common Hyperparameters
  - Number of Neighbors
  - Distance Metric (e.g., Euclidean, Manhattan)
  - Weighting Method (Uniform, Distance)

#### Decision Trees

- Assumptions
  - empty
- Common Hyperparameters
  - Max Depth
  - Min Samples Split
  - Min Samples Leaf
  - Max Features

#### Naive Bayes

- Assumptions
  - empty
- Common Hyperparameters
  - empty

### General Assumptions of ML Algorithms

- Representative Data
- Independent and Identically Distributed (i.i.d.) Data
- Sufficient Data
- Appropriate Dimensionality
- Informative and Relevant Features
- Balanced Distribution of Classes (for Classification)
- Encoded Categorical Features
- Proper Feature Scaling for Numerical Features
- Stable Feature Distributions
- Data Quality – Outliers
- Data Quality – Missing Values

### Ensemble Learning Algorithms

- Voting
  - Hard Voting
  - Soft Voting
- Stacking
- Blending
- Bagging
  - Random Forests
  - Extra Trees
- Boosting
  - Ada Boost
  - Gradient Boost
    - Standard Gradient Boost
    - XGBoost
    - LightGBM
    - CatBoost

### Unsupervised Learning Algorithms
- Clustering Algorithms
  - K-Means Clustering
  - Hierarchical Clustering
  - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
  - Gaussian Mixture Model (GMM)
- Dimensionality Reduction Algorithms
  - Principal Component Analysis (PCA)
  - t-SNE (t-Distributed Stochastic Neighbor Embedding)
- Anomaly Detection Algorithms
  - Isolation Forest


### 3.2 Regularization

- L1 (Lasso)
- L2 (Ridge)
- Elastic Net


### Machine Learning Algorithm Characteristics
  
#### General Performance

- Which ML algorithms generalize well and are resistant to overfitting?
- Which ML algorithms are prone to overfitting?
- Which ML algorithms are robust to noisy data?

#### Data Handling

- Which ML algorithms handle outliers effectively?
- Which ML algorithms require feature scaling?
- Which ML algorithms are robust to missing values?
- Which ML algorithms perform well on imbalanced datasets?

- Accuracy-Interpretability Tradeoff

#### Computational Efficiency

- Which ML algorithms require high memory?
- Which ML algorithms require long training times?
- Which ML algorithms have fast inference times?

#### Hyperparameter & Model Tuning

- Which ML algorithms require minimal hyperparameter tuning?
- Which ML algorithms are sensitive to hyperparameter selection?
- Which ML algorithms benefit the most from ensemble methods?

#### Bias-Variance Trade-off

- Which ML algorithms have low bias and high variance?
- Which ML algorithms have high bias and low variance?
- Which ML algorithms strike a good balance between bias and variance?

#### Interpretable vs. Complex Models

- Which ML algorithms are easy to interpret?
- Which ML algorithms act as "black boxes"?

#### Scalability & Deployment

- Which ML algorithms scale well with large datasets?
- Which ML algorithms are well-suited for real-time applications?

---

## Evaluations Metrics

### Regression Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R^2 or Coefficient of Determination)
- Adjusted R-squared (R^2_adj​)

### Classification Metrics

- Confusion Matrix
- Accuracy Score
- F1 Score
- Precision
- Recall (also known as Sensitivity and True Positive Rate (TPR))
- Precision-Recall Tradeoff
- Specificity (also known as True Positive Rate (TPR))
- Type I Error Rate (also known as False Positive Rate (FPR))
- Type II Error Rate (also known as False Negative Rate (FPR))
- AUC (Area Under the Curve)
- ROC (Receiver Operator Characteristic)
- AUC-ROC Curve

### Clustering Metrics

#### Internal Metrics

- Silhouette Score (also known as Silhouette Coefficient)
- Davies-Bouldin Index (DBI)
- Calinski-Harabasz Index (Variance Ratio Criteriance)

#### External Metrics

- Rand Index (RI)
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)

---

- Model Interpretability and Explainability
  - Feature Importance
  - SHAP Values (Shapley Additive Explanations)
  - LIME (Local Interpretable Model-Agnostic Explanations)
  - Explainability and Interpretability in ML

## Best Practices in ML

### 1. Ethical AI and Fairness

- Bias detection and mitigation
- Transparency in AI systems

### 2. Explainability and Interpretability

- Explainability tools (e.g., SHAP, LIME)

### 3. Data

- Federated Learning
- Techniques for privacy-preserving ML

### 4. Model Deployment

- Scaling models in production
- Monitoring and maintaining deployed models

### 5. Glossary

- Difference between Machine Learning Algorithm and Machine Learning Model
- Feature(s)
- Variable(s)
- Label(s)
- Target
- Row(s)
- Data Point(s)
- Column(s)
- Predictor Variables
- Independent Variables
- Dependent Variable
- 