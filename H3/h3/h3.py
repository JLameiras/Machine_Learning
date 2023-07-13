#h3.py
from readline import read_init_file
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

data = loadarff('kin8nm.arff')
df = pd.DataFrame(data[0])

X, y = df.drop('y', axis=1), df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
print(y_train)


ridge = Ridge(alpha=0.1)
mlpr_early_stopping = MLPRegressor(activation='tanh', hidden_layer_sizes=(10, 10), max_iter=500, random_state=0, early_stopping=True)
mlpr_not_early_stopping = MLPRegressor(activation='tanh', hidden_layer_sizes=(10, 10), max_iter=500, random_state=0)

ridge.fit(X_train, y_train)
mlpr_early_stopping.fit(X_train, y_train)
mlpr_not_early_stopping.fit(X_train, y_train)

y_pred_ridge = ridge.predict(X_test)
y_pred_mlpr_early_stopping = mlpr_early_stopping.predict(X_test)
y_pred_mlpr_not_early_stopping = mlpr_not_early_stopping.predict(X_test)

mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mae_mlpr_early_stopping = mean_absolute_error(y_test, y_pred_mlpr_early_stopping)
mae_mlpr_not_early_stopping = mean_absolute_error(y_test, y_pred_mlpr_not_early_stopping)

print('MAE linear regression = ', mae_ridge)
print('MAE MLP1 = ', mae_mlpr_early_stopping)
print('MAE MLP2 = ', mae_mlpr_not_early_stopping)

y_res_ridge = abs(y_test - y_pred_ridge)
y_res_mlpr_early_stopping = abs(y_test - y_pred_mlpr_early_stopping)
y_res_mlpr_not_early_stopping = abs(y_test - y_pred_mlpr_not_early_stopping)

plt.boxplot((y_res_ridge, y_res_mlpr_early_stopping, y_res_mlpr_not_early_stopping), labels=('Ridge', 'MLP Early Stopping', 'MLP Not Early Stopping'))
plt.ylabel('Residue (abs)')
plt.title('Boxplot Residue')
plt.show()

plt.hist((y_res_ridge, y_res_mlpr_early_stopping, y_res_mlpr_not_early_stopping))
plt.title('Histogram Residue')
plt.ylabel('Count')
plt.xlabel('Residue (abs)')
plt.show()
