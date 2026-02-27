from sklearn.metrics import mean_absolute_error
import pandas as pd

#Sellecting data for modelling
melbourne_data_path = '/Users/ninogevorkiani/Desktop/Housing-Price-Predictions/0-melb_data.csv'
melbourne_data = pd.read_csv(melbourne_data_path)

#Sellecting prediction target
melbourne_data = melbourne_data.dropna(axis=0) #removing null values
y = melbourne_data.Price

#Choosing features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
x = melbourne_data[melbourne_features]

#print(x.describe())
print('')
print(x.head())

#Building your model
from sklearn.tree import DecisionTreeRegressor

#Define Model. Specify a number of random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor()

#Fit Model
melbourne_model.fit(x,y)

#Making predictions for the following 5 houses
print('\nMaking predictions for the melbourne houses:\n')
predicted_home_prices = melbourne_model.predict(x)
print(predicted_home_prices)


#Calcxulating mean absolute error
error = mean_absolute_error(y, predicted_home_prices)
print(f'\nMean Absolute Error:{error}')

# This is the in-sample result. Therefore we will split data to use some to fit model and other to validate it  
from sklearn.model_selection import train_test_split

train_x, validation_x, train_y, validation_y = train_test_split(x, y, random_state=0)

#Define Model
melbourne_model = DecisionTreeRegressor()

#Fit Model
melbourne_model.fit(train_x, train_y)

#Get predicted prices on validation data 
validation_prediction = melbourne_model.predict(validation_x)
error = mean_absolute_error(validation_y,validation_prediction)
print(f'New Mean Absolute Error:{error}')

#Our mean absolute error for the in-sample data was about 500 dollars. Out-of-sample it is more than 250,000 dollars.