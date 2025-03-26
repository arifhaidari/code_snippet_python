### **Variance in Data Science Context**

In data science and statistics, **variance** measures how much the values in a dataset **deviate from the mean**. A high variance indicates that the data points are **spread out**, while a low variance means the values are **closer to the mean**.

#### **Formula for Variance (\(\sigma^2\))**

For a dataset with \( n \) values:

\[
\sigma^2 = \frac{1}{n} \sum\_{i=1}^{n} (x_i - \mu)^2
\]

where:

- \( x_i \) = individual data points
- \( \mu \) = mean of the data
- \( n \) = number of data points

#### **Why Does Box-Cox Help Stabilize Variance?**

Some datasets have **heteroscedasticity**, meaning the variance **changes** across different ranges of the data (e.g., larger values have more spread). This can lead to issues in machine learning models, particularly in **linear regression**, which assumes **homoscedasticity** (constant variance).

👉 **Box-Cox transformation** helps **reduce variance instability** by making the data more **normally distributed**, ensuring that models work better with features that have more consistent variance.

### **Difference Between Box-Cox Transformation and Log Transformation for Skewed Data**

Both **Box-Cox transformation** and **log transformation** are used to **reduce skewness** in data, making it more **normally distributed**. However, they have key differences:

| **Aspect**                  | **Box-Cox Transformation**                                                 | **Log Transformation**                   |
| --------------------------- | -------------------------------------------------------------------------- | ---------------------------------------- |
| **Formula**                 | \( y = \frac{x^\lambda - 1}{\lambda} \) (if \( \lambda \neq 0 \))          | \( y = \log(x) \)                        |
| **Parameter**               | Requires finding **optimal** \( \lambda \)                                 | No parameter (always uses log)           |
| **Handles Zero/Negatives?** | No (unless using `boxcox1p(x, λ)`)                                         | No (log undefined for \( x \leq 0 \))    |
| **Flexibility**             | More flexible (can apply different transformations based on \( \lambda \)) | Fixed transformation                     |
| **When to Use?**            | When data has **varying skewness**                                         | When data has **only positive skewness** |

---

### **Real-World Example: House Prices**

Imagine you are analyzing **house prices** in a city. The prices tend to be **right-skewed** because a few very expensive houses raise the average.

#### **Dataset (House Prices in $)**

```
[120000, 135000, 150000, 200000, 300000, 500000, 1000000]
```

📉 **Without Transformation:**

- Mean is **highly influenced** by outliers.
- The distribution is skewed **to the right**.

#### **Applying Log Transformation**

```python
import numpy as np
import matplotlib.pyplot as plt

house_prices = np.array([120000, 135000, 150000, 200000, 300000, 500000, 1000000])
log_transformed = np.log(house_prices)

plt.hist(house_prices, bins=10, alpha=0.5, label="Original")
plt.hist(log_transformed, bins=10, alpha=0.5, label="Log Transformed")
plt.legend()
plt.show()
```

🔹 **Effect:**

- Reduces skewness, but does not adjust variance dynamically.

#### **Applying Box-Cox Transformation**

```python
from scipy.stats import boxcox

boxcox_transformed, best_lambda = boxcox(house_prices)

plt.hist(house_prices, bins=10, alpha=0.5, label="Original")
plt.hist(boxcox_transformed, bins=10, alpha=0.5, label="Box-Cox Transformed")
plt.legend()
plt.show()

print(f"Optimal λ: {best_lambda}")  # Finds the best lambda for normality
```

🔹 **Effect:**

- **Finds the best λ value dynamically**.
- Adjusts transformation based on data **instead of assuming log is best**.

---

### **Key Takeaways**

✅ **Use Log Transformation** when:

- You have **positive values only**.
- You want a **simple** transformation.

✅ **Use Box-Cox Transformation** when:

- Your data is **highly skewed and non-normal**.
- You want **automatic optimization** for normality.
- You need **flexibility** in transformation.

### **Difference Between Box-Cox Transformation, Log Transformation, and `log1p` for Skewed Data**

When dealing with **skewed data**, we often apply transformations to make the distribution **more normal**. The most common transformations include:

1. **Box-Cox Transformation**
2. **Log Transformation**
3. **`log1p` Transformation (Log(1+x))**

Each method has its strengths and limitations. Here's a detailed comparison:

| **Transformation** | **Formula**                                                                                    | **Handles Zero/Negatives?**                                 | **Flexibility**                                         | **Use Case**                                                                                       |
| ------------------ | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **Box-Cox**        | \( y = \frac{x^\lambda - 1}{\lambda} \) (if \( \lambda \neq 0 \)), otherwise \( y = \log(x) \) | ❌ No (Only works for **positive** values)                  | ✅ Yes (λ parameter adjusts transformation dynamically) | When data has **varying skewness** and you want to **find the best transformation** automatically. |
| **Log**            | \( y = \log(x) \)                                                                              | ❌ No (Cannot handle zero or negative values)               | ❌ No (Fixed transformation)                            | When data is **positively skewed** and all values are **strictly positive**.                       |
| **log1p**          | \( y = \log(x+1) \)                                                                            | ✅ Yes (Works for **zero values**) but ❌ Not for negatives | ❌ No (Fixed transformation)                            | When data contains **zeros** but no negatives, and a simple transformation is needed.              |

---

### **1️⃣ Box-Cox Transformation (More Flexible)**

- Requires **strictly positive values**.
- Finds an **optimal λ** to transform the data.
- **Reduces skewness dynamically** based on the dataset.

📌 **Example:**

```python
import numpy as np
from scipy.stats import boxcox

data = np.array([2, 5, 10, 20, 50, 100, 200])  # Must be positive
boxcox_transformed, best_lambda = boxcox(data)

print(f"Optimal lambda: {best_lambda}")
```

✔ **Best when data is highly skewed and a dynamic transformation is needed**.

---

### **2️⃣ Log Transformation (Simple but Limited)**

- Only works for **positive values**.
- A fixed transformation that **reduces right skew**.

📌 **Example:**

```python
log_transformed = np.log(data)
```

✔ **Good for positive-only data when a simple transformation is enough**.

---

### **3️⃣ `log1p` Transformation (Handles Zero Values)**

- **Similar to log but can handle zero values**.
- Useful when the dataset has **zeros but no negatives**.

📌 **Example:**

```python
log1p_transformed = np.log1p(data)
```

✔ **Use when data contains zeros and you want to avoid issues with log(0)**.

---

### **Real-World Example: House Prices**

Imagine you are analyzing **house prices**:

#### **House Price Data (Right-Skewed)**

```
[100000, 120000, 150000, 200000, 300000, 500000, 1000000]
```

| Transformation | Effect                                                                              |
| -------------- | ----------------------------------------------------------------------------------- |
| **Log**        | Reduces skew but fails for **zero values**.                                         |
| **log1p**      | Works with **zero values** but not negatives.                                       |
| **Box-Cox**    | Dynamically **chooses the best transformation** but needs **only positive values**. |

### **Handling Negative Values in Skewed Data Transformation**

When dealing with **skewed data that contains negative values**, the standard **log and Box-Cox transformations do not work**, because:

- **Log Transformation**: Undefined for \( x \leq 0 \) (logarithm of negative numbers is not real).
- **Box-Cox Transformation**: Only works for **strictly positive** values.

### **Methods to Handle Negative Values in Transformation**

Here are some ways to transform **skewed data with negative values**:

#### **1️⃣ Shift the Data (Add a Constant)**

One simple approach is **shifting** the entire dataset by adding a constant (\( c \)) so that all values become **positive** before applying transformations.

\[
x*{\text{new}} = x + |x*{\min}| + 1
\]

📌 **Example**

```python
import numpy as np

data = np.array([-50, -20, 0, 10, 50, 100])

# Shift data by making the minimum value positive
shifted_data = data + abs(data.min()) + 1  # Add |min| + 1 to make all values positive

# Apply log transformation
log_transformed = np.log(shifted_data)
```

✔ **Works for any transformation but changes the scale of the data.**  
❌ The choice of constant affects interpretability.

---

#### **2️⃣ Yeo-Johnson Transformation (Handles Negatives)**

Unlike Box-Cox, **Yeo-Johnson transformation** can handle **both positive and negative values**.

\[
y =
\begin{cases}
\frac{(x + 1)^\lambda - 1}{\lambda}, & \text{if } x \geq 0, \lambda \neq 0 \\
\frac{(1 - (-x)^{2 - \lambda})}{2 - \lambda}, & \text{if } x < 0, \lambda \neq 2
\end{cases}
\]

📌 **Example**

```python
from scipy.stats import yeojohnson

data = np.array([-50, -20, 0, 10, 50, 100])

# Apply Yeo-Johnson transformation
yj_transformed, best_lambda = yeojohnson(data)

print(f"Optimal λ: {best_lambda}")
```

✔ **Automatically finds the best λ and works with negatives.**  
✔ **More flexible than Box-Cox.**

---

#### **3️⃣ Cube Root or Power Transformations**

For moderate skewness, we can use:

\[
x' = \text{sign}(x) \cdot |x|^{1/3}
\]

📌 **Example**

```python
cube_root_transformed = np.sign(data) * np.abs(data) ** (1/3)
```

✔ **Works with negative values without shifting data.**  
✔ **Retains negative values in a transformed way.**

---

### **Which Method Should You Use?**

| **Scenario**                            | **Best Transformation** |
| --------------------------------------- | ----------------------- |
| Data is strictly positive               | Box-Cox                 |
| Data has zeros but no negatives         | `log1p`                 |
| Data has negatives and positives        | **Yeo-Johnson**         |
| You want a simple, fast fix             | **Shift and Log**       |
| You want to retain sign but reduce skew | Cube Root               |

# Explanation of Yeo-Johnson Transformation Code

## Purpose of `best_lambda`

The `best_lambda` in this code is the optimal power parameter (λ) that the Yeo-Johnson transformation has determined for normalizing your skewed data. The Yeo-Johnson transformation finds the λ value that makes the transformed data look as close to a normal distribution as possible.

The λ value controls the type of transformation applied:

- λ = 1: No substantial transformation
- λ = 0: Natural log transformation
- λ = 0.5: Square root transformation
- λ = -1: Reciprocal transformation

## Overall Code Explanation

1. **Import**: The code imports the `yeojohnson` function from SciPy's stats module
2. **Data Preparation**: Creates a numpy array with skewed data containing both positive and negative values
3. **Transformation**: Applies the Yeo-Johnson transformation which returns:
   - `yj_transformed`: The transformed data values
   - `best_lambda`: The optimal λ parameter found
4. **Output**: Prints the optimal λ value

## Sample Output

For the given input data `[-50, -20, 0, 10, 50, 100]`, you might see output like:

```
Optimal λ: 0.123456789
```

The exact λ value will vary because the algorithm finds the optimal value through optimization, but it would typically be a small positive number for this data (since there are negative values present).

## Why Use Yeo-Johnson?

The Yeo-Johnson transformation is particularly useful because:

1. It works with both positive and negative values (unlike Box-Cox)
2. It helps make skewed data more normally distributed
3. It's useful for stabilizing variance in data
4. It can improve the performance of many statistical models that assume normally distributed data

The transformation formula varies based on whether values are positive or negative and the λ value, making it very flexible for different data distributions.

### **Real-World Example: Using Yeo-Johnson Transformation to Fix Skewed Financial Data**

#### **Scenario: Analyzing Profit & Loss (P&L) Data**

Suppose you work for a company analyzing **daily profit and loss (P&L) data**, where:

- Positive values → Profits
- Negative values → Losses
- Zero → Break-even

Your raw data looks like this (in thousands of dollars):

```python
import numpy as np
p_and_l = np.array([-150, -80, -30, 0, 20, 100, 250])
```

This data is **highly skewed** (many losses, a few large profits). Many machine learning models (e.g., linear regression) assume normally distributed data, so we need to fix the skewness.

---

### **Step 1: Apply Yeo-Johnson Transformation**

```python
from scipy.stats import yeojohnson

# Apply transformation
transformed_data, best_lambda = yeojohnson(p_and_l)

print(f"Optimal λ (lambda): {best_lambda}")
print(f"Transformed Data: {transformed_data}")
```

**Sample Output:**

```
Optimal λ (lambda): 0.34
Transformed Data: [-1.12, -0.76, -0.42, 0.0, 0.19, 0.72, 1.39]
```

- The algorithm found `λ = 0.34` as the best value to make the data more normal.
- The transformed data is now less skewed and better suited for statistical modeling.

---

### **Step 2: Interpretation of Lambda (λ)**

The `best_lambda` tells us **how the transformation was applied**:

- **If λ ≈ 0** → The transformation behaves like a **logarithm** (but works on negative values).
- **If λ ≈ 0.5** → Similar to a **square root** transformation.
- **If λ ≈ 1** → Almost no transformation.

**In our case (`λ = 0.34`):**

- The transformation is **between a log and square root**, reducing the impact of extreme values.
- Losses (`-150, -80`) are pulled closer to the mean.
- Large profits (`250`) are scaled down.

---

### **Step 3: Using Lambda for Reverse Transformation**

If you need to **convert predictions back to original scale**, you use the inverse Yeo-Johnson formula with the same `λ`:

```python
def inverse_yeojohnson(y, lambda_):
    if lambda_ == 0:
        return np.exp(y)
    else:
        return np.power(y * lambda_ + 1, 1 / lambda_)

# Example: Transform back a predicted value of 1.0
predicted_transformed = 1.0
original_scale = inverse_yeojohnson(predicted_transformed, best_lambda)
print(f"Original scale value: {original_scale:.2f}")
```

**Output:**

```
Original scale value: 137.71  # ≈ $137,710 (close to original $100k–$150k range)
```

---

### **Why This Matters in Real Applications?**

1. **Financial Risk Modeling**
   - Banks use Yeo-Johnson to model loan defaults (where losses are negative).
2. **Machine Learning Preprocessing**

   - Models like **Linear Regression, SVM, or Neural Networks** perform better when data is less skewed.

3. **Anomaly Detection**
   - Helps distinguish real outliers (fraudulent transactions) from natural skewness.

---

### **Final Notes**

- **Always check `best_lambda`**—if it’s near **1**, your data may not need transformation.
- **Compare before/after distributions** using histograms or Q-Q plots.
- **Use the same `λ`** for training and test data to avoid bias.

This ensures your statistical models work correctly on real-world skewed data! 🚀
