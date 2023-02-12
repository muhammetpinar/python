import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

linear_regression = LinearRegression()
linear_regression.fit(xtrain, ytrain)

import pickle
with open("pickle_model", "wb") as file:
    pickle.dump(linear_regression, file)


with open("pickle_model", "rb") as file:
    model = pickle.load(file)

predictions = model.predict(xtest)

for i in range(len(predictions)):
    print(predictions[x], xtest[x], [ytest[x]])