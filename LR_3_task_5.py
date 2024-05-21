import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

m = 100
X = np.linspace(-3, 3, m)
y = 2 * np.sin(X) + np.random.uniform(-0.6, 0.6, m)

plt.scatter(X, y, color='blue', label='Дані')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Випадкові дані (Варіант 6)')
plt.legend()
plt.show()

lin_reg = LinearRegression()
X_reshaped = X.reshape(-1, 1)
lin_reg.fit(X_reshaped, y)

plt.scatter(X, y, color='blue', label='Дані')
plt.plot(X, lin_reg.predict(X_reshaped), color='red', label='Лінійна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Лінійна регресія (Варіант 6)')
plt.legend()
plt.show()

degree = 10
poly_reg = PolynomialFeatures(degree=degree)
X_poly = poly_reg.fit_transform(X_reshaped)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

plt.scatter(X, y, color='blue', label='Дані')
plt.plot(X, lin_reg_2.predict(X_poly), color='green', label=f'Поліноміальна регресія (ступінь {degree})')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Поліноміальна регресія (Варіант 6, ступінь {degree})')
plt.legend()
plt.show()

mse = mean_squared_error(y, lin_reg_2.predict(X_poly))
print("Середньо-квадратична помилка (MSE) поліноміальної регресії:", mse)
