import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import pickle

# Завантаження даних
input_file = 'data_regr_1.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розділення даних на тренувальні та тестові набори
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# Створення та тренування моделі
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# Прогноз на тестовому наборі
y_test_pred = regressor.predict(X_test)

# Візуалізація результатів
plt.scatter(X_test, y_test, color='green', label='Фактичні дані')
plt.plot(X_test, y_test_pred, color='black', linewidth=2, label='Прогноз')
plt.xlabel('Вхідна змінна')
plt.ylabel('Цільова змінна')
plt.legend()
plt.show()

# Оцінка якості моделі
print("Linear regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# Збереження моделі у файл
output_model_file = 'model.pkl'
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)

# Завантаження моделі з файлу
with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)

# Повторний прогноз з використанням завантаженої моделі
y_test_pred_new = regressor_model.predict(X_test)
print("\nNew mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))
