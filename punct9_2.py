import numpy as np
import pandas as pd

# Визначення матриці A
A = np.array([
    [0.1, 0.2, 0.4],
    [0.3, 0.2, 0.3],
    [0.1, 0.3, 0.2]
])

# Вектор доданої вартості в цінах
s = np.array([0.4, 0.1, 0.3])

# Одинична матриця E
E = np.eye(3)

# Власні числа і вектори матриці A
eigenvalues_A, eigenvectors_A = np.linalg.eig(A)

# Транспонування матриці A для знаходження лівих векторів
AT = A.T
eigenvalues_AT, eigenvectors_AT = np.linalg.eig(AT)

# Визначення характеристичного поліному
char_poly = np.poly(A)

# Визначення числа Фробеніуса
frobenius_number = np.max(np.abs(eigenvalues_A.real))

# Власні вектори відповідно до числа Фробеніуса
frobenius_vector_A = eigenvectors_A[:, np.argmax(np.abs(eigenvalues_A.real))]
frobenius_vector_AT = eigenvectors_AT[:, np.argmax(np.abs(eigenvalues_AT.real))]

# Розрахунок матриці повних витрат B
B = np.linalg.inv(E - A)

# Розрахунок вектора цін
price_vector = np.dot(B.T, s)

# Функція для обчислення ступенів матриці та їх різниць
def matrix_powers_and_differences(A, max_power, convergence_threshold):
    powers = [E]
    differences = []
    current_power = E
    sum_series = np.copy(E)
    for n in range(1, max_power + 1):
        previous_power = current_power
        current_power = np.dot(current_power, A)
        powers.append(current_power)
        differences.append(current_power - previous_power)
        sum_series += current_power
        if np.max(np.abs(differences[-1])) < convergence_threshold:
            break
    return powers, differences, sum_series, n

# Виконання обчислень
max_power = 20
convergence_threshold = 0.01
powers, differences, sum_series, n_converged = matrix_powers_and_differences(A, max_power, convergence_threshold)

# Формування DataFrame для відображення результатів
data = {}
for i in range(len(powers)):
    data[f"A^{i}"] = powers[i].flatten()
    if i < len(differences):
        data[f"A^{i+1}-A^{i}"] = differences[i].flatten()
data[f"Sum Series (Converged at n={n_converged})"] = sum_series.flatten()

# Результати обчислень
df_results = pd.DataFrame(data)
df_results = df_results.T  # Транспонування для кращого відображення

# Результати для виводу
eigenvalues_info = {
    'Eigenvalues': eigenvalues_A,
    'Frobenius Left Vector (from A^T)': frobenius_vector_AT,
    'Characteristic Polynomial': char_poly,
    'Frobenius Number': frobenius_number,
    'Full Cost Matrix B': B,
    'Price Vector p': price_vector
}

# Виведення результатів у консоль
print("Eigenvalues and Eigenvectors:\n")
for key, value in eigenvalues_info.items():
    if isinstance(value, np.ndarray):
        print(f"{key}:\n{value}\n")
    else:
        print(f"{key}: {value}\n")

print("Matrix Powers and Differences:")
print(df_results)
