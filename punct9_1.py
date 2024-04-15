import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# дані для поточного року
data = {
    "Sector": ["Industry", "Agriculture"],
    "Industry Demand (millions UAH)": [400, 100],
    "Agriculture Demand (millions UAH)": [300, 200],
    "Population Demand (millions UAH)": [900, 500]
}

# Створення DataFrame
df_current_year = pd.DataFrame(data)

# Обрахунок загального випуску для кожної галузі
df_current_year["Total Output (millions UAH)"] = df_current_year.sum(axis=1, numeric_only=True)

# Потреба на наступний рік згідно нового попиту
new_demand_next_year = {
    "Industry": 850,
    "Agriculture": 700
}

# Коректний обрахунок загального випуску на наступний рік
required_output_next_year = {
    "Industry": 400 + 300 + new_demand_next_year["Industry"],
    "Agriculture": 100 + 200 + new_demand_next_year["Agriculture"]
}

# Технологічна матриця A
A = np.array([
    [0.25, 0.375],
    [0.0625, 0.25]
])

# Одинична матриця E
E = np.eye(2)

# Розрахунок матриці E - A
E_minus_A = E - A

# Обрахунок матриці повних затрат S
S = np.linalg.inv(E_minus_A)

# Вивід результатів матричних обчислень
print("Matrix A (Technology Matrix):\n", A)
print("Matrix E - A:\n", E_minus_A)
print("Matrix S (Total Requirements Matrix):\n", S)

# Результати для візуалізації
results = pd.DataFrame({
    "Sector": ["Industry", "Agriculture"],
    "Total Output This Year (millions UAH)": df_current_year["Total Output (millions UAH)"],
    "Required Output Next Year (millions UAH)": [required_output_next_year["Industry"],
                                                required_output_next_year["Agriculture"]]
})

# Візуалізація результатів
fig, ax = plt.subplots()
results.set_index("Sector").plot(kind='bar', ax=ax)
ax.set_title("Comparison of Total Output Requirements")
ax.set_ylabel("Output (millions UAH)")
ax.set_xlabel("Sector")
plt.show()

# Виведення таблиці поточного року та прогнозу на наступний рік
print("Current Year Output Data:")
print(df_current_year)
print("\nProjected Requirements for Next Year:")
print(results)
