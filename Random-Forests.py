import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

iowa_file_path = '/Users/ninogevorkiani/Desktop/Housing-Price-Predictions/csv-iowa.csv'
iowa_data = pd.read_csv(iowa_file_path)

#Target Object
y = iowa_data.SalePrice
#Features
features =  ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = iowa_data[features]

#Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)

#Fit Model
iowa_model.fit(train_X, train_y)

#Val predictions and val mae
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_y, val_predictions)
print(f'Validation MAE when leaf nodes not specified: {val_mae:.0f}')

#Using best value for leafs specified
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

#Fit Model
iowa_model.fit(train_X, train_y)

#Val predictions and val mae
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_y, val_predictions)
print(f'Validation MAE when leaf nodes are specified: {val_mae:.0f}')

