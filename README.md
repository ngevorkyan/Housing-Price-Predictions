# Housing-Price-Predictions

This project marks my first step into Machine Learning, guided by Kaggle's tutorials and hands-on exercises. It focuses on predicting house prices using decision tree models.

Datasets:
The project uses two datasets containing housing data:
Iowa Housing Dataset – includes various features of homes in Iowa.
Melbourne Housing Dataset – includes housing data from Melbourne, Australia.

Each dataset includes a target variable (y) representing the house price, which we aim to predict.

Project Workflow:
1. Feature Selection 
From each dataset, relevant features are selected to build the predictive model.
2. Modeling
A Decision Tree Regressor is created to predict house prices.
3. Training and Validation
The model is fit on the training data.

Its performance is evaluated on the validation set using Mean Absolute Error (MAE).

Model Tuning:
Different tree depths are tested to find the one that yields the best MAE.
The model is adjusted to avoid underfitting or overfitting depending on MAE results.

Objective:
Learn the basics of Machine Learning workflow.
Understand how Decision Trees can be used for regression tasks.
Evaluate model performance using MAE and interpret results.
