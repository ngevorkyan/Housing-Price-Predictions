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

#Mean absolute error
from sklearn.metrics import mean_absolute_error

print('\nValidation MAE:', mean_absolute_error(validation_y,predicted_with_validation_data) , '\n')

#Underfitting vs overfitting

def get_mae(max_leaf_nodes, train_x, validation_x, train_y, validation_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_x, train_y)
    predictions = model.predict(validation_x)
    mae = mean_absolute_error(validation_y, predictions)
    return mae



candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

# Store the results
results = []

for leaf in candidate_max_leaf_nodes:
    # Compute MAE for the current leaf count
    my_mae = get_mae(leaf, train_x, validation_x, train_y, validation_y)
    
    # Store leaf and MAE as a tuple
    results.append((leaf, my_mae))

    # Print progress neatly
    print(f"Leaf nodes: {leaf}\tMean Absolute Error: {my_mae:.2f}")

# Find the leaf count with the minimum MAE
best_leaf, best_mae = min(results, key=lambda x: x[1])

print(f"\nBest leaf count: {best_leaf} with MAE: {best_mae:.2f}")