import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('dataset12')

x = data[['Usia Mobil (tahun)']]
y = data[['Harga Mobil ($100)']]

model = LinearRegression()

model.fit(X, y)

print('Koefisien (slope):', model.coef_[0])
print('Intercept:', model.intercept_)
