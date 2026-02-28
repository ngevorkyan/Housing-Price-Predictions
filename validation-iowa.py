import pandas as pd
from sklearn.model_selection import train_test_split

iowa_file_path = '/Users/ninogevorkiani/Desktop/Housing-Price-Predictions/csv-iowa.csv'
iowa_data = pd.read_csv(iowa_file_path)
print(iowa_data.shape)


#Choose Features
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
x = iowa_data[features]

#Choose Anaysis Target
y = iowa_data.SalePrice

#Build a model
from sklearn.tree import DecisionTreeRegressor
iowa_model = DecisionTreeRegressor(random_state=0)

#Fit model
iowa_model.fit(x,y)

print('First in-sample predictions.   :', iowa_model.predict(x.head()))
print('Actual target values           :', y.head().tolist())


#Split the data
train_x, validation_x, train_y, validation_y = train_test_split(
    x, y, random_state=0
)

#Build a model
iowa_model = DecisionTreeRegressor(random_state=0)

#Fit model with just train data
iowa_model.fit(train_x, train_y)

#Predict with both datas
predicted_with_train_data = iowa_model.predict(train_x)
predicted_with_validation_data = iowa_model.predict(validation_x)
print('Predicted with only train data :',predicted_with_train_data)
print('Predicted with validation data :',predicted_with_validation_data)

#Mean absolute error
from sklearn.metrics import mean_absolute_error

print('Validation MAE:', mean_absolute_error(validation_y,predicted_with_validation_data))

