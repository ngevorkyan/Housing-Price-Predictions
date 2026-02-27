import pandas as pd


#Sellecting data for modelling
melbourne_data_path = '/Users/ninogevorkiani/Desktop/Housing-Price-Predictions/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_data_path)



#Sellecting prediction target
melbourne_data = melbourne_data.dropna(axis=0) #removing null values
y = melbourne_data.Price

#Choosing features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x = melbourne_data[melbourne_features]

#print(x.describe())
print('')
print(x.head())

#Building your model
from sklearn.tree import DecisionTreeRegressor

#Define Model. Specify a number of random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

#Fit Model
melbourne_model.fit(x,y)

#Making predictions for the following 5 houses
print('\nMaking predictions for the following 5 houses:/n')
print(melbourne_model.predict(x.head()))