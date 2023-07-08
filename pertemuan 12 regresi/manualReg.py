import pandas as pd

data = pd.read_csv('dataset12')

n = len(data)


sum_xy = (data['Usia Mobil (tahun)'] * data['Harga Mobil ($100)']).sum()
sum_x = data['Usia Mobil (tahun)'].sum()
sum_y = data['Harga Mobil ($100)'].sum()
sum_x2 = (data['Usia Mobil (tahun)'] ** 2).sum()


m = ((sum_xy - (sum_x * sum_y) / n) /
     (sum_x2 - (sum_x ** 2) / n))
c = (sum_y - m * sum_x) / n


print('Koefisien (slope):', m)
print('Intercept:', c) 