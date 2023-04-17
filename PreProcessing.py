import pandas as pd
import numpy as np


data = pd.read_csv('dataSetTugasDataMining.csv')
X = data.iloc[:, :-1].values
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Drop kolom yang tidak diperlukan
data = data.drop(['Jumlah Kunjungan'], axis=1)

# Encode variabel kategorikal menggunakan one-hot encoding
data_encoded = pd.get_dummies(data, columns=['Jenis Kelamin', 'Lokasi Geografis', 'Jenis Produk'])

# Normalisasi variabel numerik menggunakan MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_encoded[['Usia Pelanggan', 'Jumlah Pembelian']] = scaler.fit_transform(data_encoded[['Usia Pelanggan', 'Jumlah Pembelian']])

# Pisahkan variabel target
X = data_encoded.drop(['Keberhasilan Pemasaran'], axis=1)
y = data_encoded['Keberhasilan Pemasaran']

print(data_encoded[['Usia Pelanggan', 'Jumlah Pembelian']].describe())