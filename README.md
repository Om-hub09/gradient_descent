Batch Gradient Descent (BGD) is a foundational optimization algorithm used in machine learning to minimize a model's cost function. It is also known simply as vanilla gradient descent. The core defining characteristic of Batch Gradient Descent is that it calculates the gradient of the cost function using the entire training dataset before performing a single parameter update. This process is repeated iteratively over multiple training epochs until the model converges to an optimal solution

## Diabetes Progression Prediction using Linear Regression & Gradient Descent

This project implements and evaluates multiple linear regression models on the **Scikit-learn Diabetes Dataset** to predict disease progression. It contrasts a standard Ordinary Least Squares (OLS) framework against a custom-built Batch Gradient Descent solver from scratch.

### 📋 Project Overview
1. **Dataset Exploration:** Loaded the Diabetes dataset consisting of **442 samples** and **10 baseline feature variables** (e.g., age, sex, body mass index, average blood pressure).
2. **Data Preprocessing:** Segmented the data into training and testing splits utilizing an **80/20 ratio**.
3. **Scikit-Learn Benchmark:** Built an OLS baseline model using Scikit-learn's `LinearRegression` class, yielding an R² score of approximately **0.44**.
4. **Custom Optimization Implementation:** Engineered a custom `GDRegressor` Python class from scratch to mathematically simulate multi-variable **Batch Gradient Descent**.

### 🛠️ Technical Workflow & Implementation

#### 1. Baseline Model (Scikit-Learn)
* **Model:** `LinearRegression()`
* **Performance Metric:** R² ≈ 0.44 
* **Extracted Weights:** Intercept ≈ 151.88 with a 10-dimensional coefficient vector.

#### 2. Custom Batch Gradient Descent (`GDRegressor`)
An iterative optimization class built solely on top of NumPy arrays featuring explicit derivative equations for vectorized coefficient and intercept updating loops.

```python
class GDRegressor:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.coef_ = None
        self.intercept_ = None
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, x_train, y_train):
        self.intercept_ = 0
        self.coef_ = np.ones(x_train.shape[1])

        for i in range(self.epochs):
            y_hat = np.dot(x_train, self.coef_) + self.intercept_
            
            # Derivative calculation & weight adjustments
            intercept_der = -2 * np.mean(y_train - y_hat)
            self.intercept_ -= (self.learning_rate * intercept_der)

            coef_der = -2 * np.dot((y_train - y_hat), x_train) / x_train.shape[1]
            self.coef_ -= (self.learning_rate * self.coef_) # NOTE: Ongoing optimization required here

    def predict(self, x_test):
        return np.dot(x_test, self.coef_) + self.intercept_
```

### 🔍 Current Findings & Next Steps
* The standard Scikit-learn model captures roughly 44% of the target's variance. 
* The scratch-built `GDRegressor` successfully converges on a comparable intercept (≈ 150.46). However, the current matrix dot-product step for updating coefficients (`coef_der`) is decaying exponentially due to an implementation error in line 22 (`self.coef_ - (self.learning_rate * self.coef_)`), which bypasses the computed gradient step vector entirely. Fixing this vector calculation remains an immediate next step to properly align the custom model's R² score with the baseline OLS performance.
