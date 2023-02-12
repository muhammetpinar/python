import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import regression

sns.set()
plt.style.use('seaborn-whitegrid')

data = pd.read_csv("ADESE.csv")
print(data.head())

data.dropna()
plt.figure(figsize=(10, 4))
plt.title("Price ")
plt.xlabel("Date")
plt.ylabel("Price")
plt.plot(data["Price"])
plt.show()

from autots import AutoTS
model = AutoTS(forecast_length=1, frequency='infer', ensemble='simple', drop_data_older_than_periods=200)
model = model.fit(data, date_col='Date', value_col='Price', id_col=None)

prediction = model.predict()
forecast = prediction.forecast
print(" Price Prediction")
print(forecast)