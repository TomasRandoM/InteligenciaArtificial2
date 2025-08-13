import kagglehub
import pandas as panda
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

dataset = panda.read_csv("C:/Users/tomas/.cache/kagglehub/datasets/nikhil7280/student-performance-multiple-linear-regression/versions/1/Student_Performance.csv")
dataset = dataset.replace({"Yes": 1, "No": 0})
dataset = dataset.infer_objects(copy=False)
datasetX = dataset[["Hours Studied","Previous Scores","Extracurricular Activities","Sleep Hours","Sample Question Papers Practiced"]]
datasetY = dataset["Performance Index"]

xTrain, xTest, yTrain, yTest = train_test_split(datasetX, datasetY, test_size=0.2, random_state=2025)

model = LinearRegression()
model.fit(xTrain, yTrain)

prediction = model.predict(xTest)

print("Coeficientes encontrados:", model.coef_)
print("MSE ", mean_squared_error(yTest, prediction))
print("MAE ", mean_absolute_error(yTest, prediction))


dataset = panda.read_csv("C:/Users/tomas/.cache/kagglehub/datasets/taseermehboob9/marketing-campaigns-logistic-regression/versions/1/Marketingcampaigns.csv")
dataset = panda.get_dummies(dataset, columns=["Location"], drop_first=True)
dataset = dataset.replace({"True": 1, "False": 0})
dataset = dataset.infer_objects(copy=False)
datasetX = dataset[["Age","Gender","Location_Brisbane", "Location_Perth", "Location_Sydney","Email Opened","Email Clicked","Product page visit","Discount offered"]]
datasetY = dataset["Purchased"]
xTrain, xTest, yTrain, yTest = train_test_split(datasetX, datasetY, test_size=0.2, random_state=2025)

model = LogisticRegression()
model.fit(xTrain, yTrain)

prediction = model.predict(xTest)

print("Coeficientes encontrados:", model.coef_)
print("MSE ", mean_squared_error(yTest, prediction))
print("MAE ", mean_absolute_error(yTest, prediction))