import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Baca data dari file CSV
data = pd.read_csv('data5.csv')

# Tentukan variabel independen (X) dan dependen (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Inisialisasi model Naive Bayes
model = GaussianNB()

# Latih model dengan data
model.fit(X, y)

# Data baru yang akan diklasifikasikan
X_new = [[1, 1, 0, 1]]

# Lakukan klasifikasi pada data baru
prediction = model.predict(X_new)

# Cetak hasil klasifikasi
print(prediction)
