import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Вхідні дані для поточного року
data = {
    "Sector": ["Industry", "Agriculture"],
    "Industry Demand (millions UAH)": [400, 100],
    "Agriculture Demand (millions UAH)": [300, 200],
    "Population Demand (millions UAH)": [900, 500]
}

# Створення DataFrame
df_current_year = pd.DataFrame(data)

# Обрахунок загального випуску для кожної галузі (лише числові колонки)
df_current_year["Total Output (millions UAH)"] = df_current_year.iloc[:, 1:].sum(axis=1)

# Технологічна матриця A
A = np.array([
    [0.25, 0.375],
    [0.0625, 0.25]
])

# Одинична матриця E
E = np.eye(2)

# Матриця E - A
E_minus_A = E - A

# Матриця повних затрат S
S = np.linalg.inv(E_minus_A)

# Зміна попиту Y
Y = np.array([850, 700])

# Розрахунок повного випуску X за формулою X = S^-1 * Y
S_inv = np.linalg.inv(S)
X = np.dot(S_inv, Y)

# Вивід результатів
print("Matrix A (Technology Matrix):\n", A)
print("Matrix E - A:\n", E_minus_A)
print("Matrix S (Total Requirements Matrix):\n", S)
print("Change in Demand (Y):\n", Y)
print("Full Production Output (X):\n", X)

# Візуалізація результатів
fig, ax = plt.subplots()
results = pd.DataFrame({
    "Sector": ["Industry", "Agriculture"],
    "Total Output This Year (millions UAH)": df_current_year["Total Output (millions UAH)"],
    "Full Production Output Next Year (X)": X
})
results.set_index("Sector").plot(kind='bar', ax=ax)
ax.set_title("Comparison of Output Requirements")
ax.set_ylabel("Output (millions UAH)")
ax.set_xlabel("Sector")
plt.show()
