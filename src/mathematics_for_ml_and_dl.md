# Complete Mathematics For Machine Learning and Deep Learning Curriculum

## 1. Linear Algebra

- Introduction to Linear Algebra
- What is Linear Algebra?
- Applications and Relevance of Linear Algebra in Machine Learning and Deep Learning
    - Linear Regression (Normal Equation & Least Squares Solution)
    - Principal Component Analysis (PCA) & Dimensionality Reduction
    - Neural Networks (Tensors, Weights as Matrices)
    - Computer Vision (Image as a Matrix, Convolutions)
    - Markov Chains & Stochastic Matrices
- Vectors
- Vector Operations
- Addition & Subtraction
- Scalar Multiplication
- Dot Product (Scalar Product) (Mention how Dot Product is related to Inner Product to avoid future confusions)
- Cross Product
- Projection of Vectors
- Angle Between Vectors & Cosine Similarity
- Vector Spaces
    - Definition of Vector Space
    - Basis and Dimension
    - Span of Vectors
    - Linear Dependence and Linear Independence
    - Change of Basis
    - Vector Projections and Cosine Similarity
- Matrices
- Matrix Operations
    - Addition & Subtraction
    - Matrix Multiplication (Row by Column, Element-wise)
    - Inverse
    - Transpose
    - Matrix Factorization
    - Singular Value Decomposition (SVD)
- Special Matrices
    - Identity Matrix
    - Diagonal Matrix
    - Symmetric Matrix
    - Orthogonal Matrix
    - Triangular Matrices (Upper & Lower)
    - Positive Definite & Positive Semi-Definite Matrices
    - Singular & Non-Singular Matrices
    - Sparse & Dense Matrices
- Norms and Distance Metrics
- Vector Norms
    - Norm (also known as Cardinality or L_0)
    - Euclidean Norm (L_2)
    - Manhattan Norm (L_1)
    - Max Norm (also known as Infinity Norm or L_∞)
    - p-Norm (L_p)
- Matrix Norms
    - Frobenius Norm
    - Spectral Norm
- Distance Metrics
    - Euclidean Distance
    - Manhattan Distance
    - Minkowski Distance
    - Cosine Similarity & Distance
- Determinants
- Properties of Determinants
- Determinant of 2×2 and 3×3 Matrices
- Cofactor Expansion (Laplace Expansion)
- Determinants & Invertibility
- Eigenvalues and Eigenvectors
- Definition & Properties
- Eigenvalue Decomposition (also known as Eigen Decomposition)
- Diagonalization of a Matrix
- Power Method for Finding Eigenvalues
- Singular Value Decomposition (SVD)
- Systems of Linear Equations
- Solving Systems Using Matrices
- Gaussian Elimination & Row Echelon Form (REF)
- Gauss-Jordan Elimination & Reduced Row Echelon Form (RREF)
- Cramer's Rule
- Matrix Inversion Method for Solving Equations
- Linear Transformations
- Definition of a Linear Transformation
- Matrix Representation of Linear Transformations
- Rotation, Scaling, Reflection, Projection, and Shear Transformations
- Kernel & Image of a Transformation
- Tensors
- Tensor Addition & Subtraction
- Tensor Multiplication (Outer Product, Tensor Contraction)
- Rank of a Tensor
- Tensors in Machine Learning & Deep Learning

---

## 2. Calculus

- Limits
- Differential Calculus
- Inferential Calculus
  - Limits
  - Derivatives
  - Chain Rule

## 3. Probability Theory

- Random Variables
  - Discrete
  - Continuous
  - Probability Density Function (PDF)
  - Probability Mass Function (PMF)
  - Cumulative Distribution Function (CDF)
  - Normalization
  - Support

### Distributions

#### Continuous Distributions

- Continuous Uniform Distribution
- Normal (Gaussian) Distribution
  - Standard Normal Distribution
  - Log-Normal Distribution
- Beta Distribution
- Gamma Distribution
- Exponential Distribution

#### Discrete Distributions

- Discrete Uniform Distribution
- Bernoulli Distribution
- Binomial Distribution
- Multinomial Distribution
- Poisson Distribution

#### Specialized Distributions

You can research and learn about these optional distributions when needed since they are only used in specific scenarios.

- Continuous Distributions
  - Dirichlet Distribution
  - Chi-Square Distribution
  - Student's t-Distribution
  - Weibull Distribution
  - Cauchy Distribution
  - F-Distribution
  - Pareto Distribution
  - Laplace Distribution
  - Rayleigh Distribution
- Discrete Distributions
  - Geometric Distribution
  - Negative Binomial Distribution

I have mentioned 23 probability distributions here. These are the most commonly used probability distributions in Machine Learning. But, there are infinitely many probability distributions. Because they are mathematical constructs used to model randomness and uncertainty, and they can be defined in a wide variety of forms depending on their specific properties and contexts. Thus, the number of probability distributions is theoretically infinite, as they are defined by functions and parameters that can take on a limitless range of forms.

---

## 4. Statistics

### Descriptive Statistics

- Measures of Central Tendency
- Measures of Dispersion
  - Variance
  - Standard Deviation
  - Standard Error
- Measures of Relatedness
  - Covariance
  - Correlations:
    - Pearson
    - Spearman
    - Kendall

### Inferential Statistics

### Bayesian Statistics

# Statistical Measures Overview

This document outlines various statistical measures, categorized into their primary groups, providing a foundation for data exploration and analysis.

## 1. **Measures of Central Tendency** (Location or Average)
- **Mean**: The arithmetic average of a dataset.
- **Median**: The middle value when data is ordered.
- **Mode**: The most frequently occurring value(s).

## 2. **Measures of Dispersion** (Spread or Variability)
- **Range**: The difference between the maximum and minimum values.
- **Interquartile Range (IQR)**: The range of the middle 50% of data (Q3 - Q1).
- **Variance**: The average squared deviation from the mean.
- **Standard Deviation**: The square root of the variance.
- **Coefficient of Variation (CV)**: The ratio of the standard deviation to the mean, often expressed as a percentage.

## 3. **Measures of Association** (Relationships between Variables)
- **Covariance**: Indicates the direction of the relationship between two variables.
- **Correlation**: A standardized measure of association (ranges from -1 to +1).
- **Spearman’s Rank Correlation**: Non-parametric correlation measure based on rank order.
- **Pearson’s Correlation Coefficient**: Linear correlation coefficient for continuous data.

## 4. **Measures of Shape** (Data Distribution Characteristics)
- **Skewness**: Measures the asymmetry of the data distribution.
- **Kurtosis**: Measures the "tailedness" of the distribution (e.g., flat vs. peaked).
- **Moments**: Higher-order moments (e.g., third and fourth) describe skewness and kurtosis.

## 5. **Outliers and Extreme Values**
- **Z-Score**: Standardized value showing how far a data point is from the mean.
- **Percentiles**: The values below which a percentage of data falls (e.g., 25th, 50th, 75th percentiles).
- **Quartiles**: Dividing data into four equal parts.

## 6. **Frequency Measures**
- **Frequency Distribution**: A table or chart showing how data values are distributed.
- **Cumulative Frequency**: The running total of frequencies.

## 7. **Other Measures in Data Analysis**
- **Entropy**: A measure of randomness or uncertainty in a dataset.
- **Moments**: Generalized measures that include:
  - Mean (1st moment)
  - Variance (2nd moment)
  - Skewness (3rd moment)
  - Kurtosis (4th moment)

---

These statistical measures form the core of **Descriptive Statistics**, enabling quantitative summaries and insights from datasets.

---

## 5. Information Theory

- KL Divergence

---

## 6. Optimization Theory

- Gradient Descent

---
