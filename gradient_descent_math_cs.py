
import numpy as np
import matplotlib.pyplot as plt

# === Step 1: Data ===
math = np.array([92, 56, 88, 70, 80, 49, 65, 35, 66, 67])
cs = np.array([98, 68, 81, 80, 83, 52, 66, 30, 68, 73])

# === Step 2: Correlation ===
correlation = np.corrcoef(math, cs)[0, 1]
print("Correlation between Math and CS:", round(correlation, 3))

# === Step 3: Gradient Descent ===
m = 0
b = 0
learning_rate = 0.0005
epochs = 10000
threshold = 0.0001

previous_cost = float('inf')

for i in range(epochs):
    predictions = m * math + b
    error = cs - predictions
    cost = np.mean(error ** 2)

    if abs(previous_cost - cost) < threshold:
        print(f"Stopping at iteration {i}, cost change below threshold.")
        break

    previous_cost = cost

    dm = -2 * np.mean(math * error)
    db = -2 * np.mean(error)

    m -= learning_rate * dm
    b -= learning_rate * db

    if i % 500 == 0:
        print(f"Step {i}: m = {m:.4f}, b = {b:.4f}, cost = {cost:.4f}")

print(f"\nFinal model: CS = {m:.2f} * Math + {b:.2f}")

# === Step 4: Plot ===
plt.scatter(math, cs, label='Actual')
plt.plot(math, m * math + b, color='red', label='Predicted Line')
plt.xlabel('Math')
plt.ylabel('CS')
plt.title('Linear Regression using Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()
