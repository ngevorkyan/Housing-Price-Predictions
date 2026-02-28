import pandas as pd

#Sellecting the data for modelling
iowa_file_path = '/Users/ninogevorkiani/Desktop/Housing-Price-Predictions/csv-iowa.csv'
iowa_data = pd.read_csv(iowa_file_path)
print(iowa_data.head())

#Specify prediction target
print(iowa_data.columns)
iowa_data = iowa_data.dropna(axis=1) #dropping na values
y = iowa_data.SalePrice

#Specify features
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
x = iowa_data[features]

#Build a model
from sklearn.tree import DecisionTreeRegressor
iowa_model = DecisionTreeRegressor(random_state=1)

#Fit model
iowa_model.fit(x,y)

#Making a prediction
print('\nMaking a prediction for following 5 houses:\n')
print(x.head())
print(f'The predicted prices are: {iowa_model.predict(x.head())}')









