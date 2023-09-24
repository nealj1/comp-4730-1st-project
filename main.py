# Choose a dataset with at least 5000 instances and 20 attributes for classification or regression. Compare how the different approaches seen in class perform on this dataset to predict accurately the classes or the values of the unlabeled data. You should determine what are the best hyper-parameters for each approach you are using. 

# MODULES ----------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
import printing as dataprint

# GLOBAL VARIABLES -------------------------------------------------------------
X_grid_size = 17
y_grid_size = 17
power_output = 17
filenames = ["Adelaide_Data","Perth_Data", "Sydney_Data", "Tasmania_Data"]
# Define the hyperparameter grids for each model
param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
}

# Create a dictionary of models and their respective parameter grids
models_and_params = {
    'Linear Regression': (LinearRegression(), {}),
    'Ridge Regression': (Ridge(), param_grid),
    'Lasso Regression': (Lasso(), param_grid),
    'Decision Tree': (DecisionTreeRegressor(), {'max_depth': [1, 5, 10, 15,20, None]}),
    'K-Nearest Neighbors': (KNeighborsRegressor(), {'n_neighbors': [1, 3, 5, 7, 10, 15]}),
    #'MLP': (MLPRegressor(), {'hidden_layer_sizes': [(50, 25), (100, 50), (100, 100)], 'alpha': [0.0001, 0.001, 0.01], 'learning_rate_init': [0.001, 0.01, 0.1]}),
}

# FUNCTIONS --------------------------------------------------------------------

# Train and evaluate a set of regression models individually on training and testing sets.
def train_and_evaluate_individual_models(X_train, y_train, X_test, y_test, models, model_names):

    model_metrics = []  # List to store metrics for each model

    for i, model in enumerate(models):
        # Train and evaluate the model on the training set
        y_pred_train = model.predict(X_train)
        mse_train = mean_squared_error(y_train, y_pred_train)
        rmse_train = np.sqrt(mse_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)

        # Train and evaluate the model on the testing set
        y_pred_test = model.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)

        # Store metrics in a dictionary for this model
        model_metrics.append({
            'Model Name': model_names[i],
            'Training MSE': mse_train,
            'Training RMSE': rmse_train,
            'Training MAE': mae_train,
            'Training R2': r2_train,
            'Testing MSE': mse_test,
            'Testing RMSE': rmse_test,
            'Testing MAE': mae_test,
            'Testing R2': r2_test,
        })

        print(f"{model_names[i]} - Training MSE: {mse_train:.2f}, RMSE: {rmse_train:.2f}, MAE: {mae_train:.2f}, R2: {r2_train:.2f}")
        print(f"{model_names[i]} - Testing MSE: {mse_test:.2f}, RMSE: {rmse_test:.2f}, MAE: {mae_test:.2f}, R2: {r2_test:.2f}")

    return model_metrics


# Perform hyperparameter tuning and evaluation for a set of regression models.
def hyperparameter_tuning_and_evaluation(X_train, y_train, X_test, y_test, models_and_params):
    print(f"\nHyperparameters")
    for model_name, (model, param_grid) in models_and_params.items():
        grid_search = GridSearchCV(model, param_grid, cv=5)#, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_params = grid_search.best_params_
        print(f"Best Hyperparameters for {model_name}: {best_params}")

        # Get the best model with the tuned hyperparameters
        best_model = grid_search.best_estimator_

        # Evaluate the best model on the test set
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"{model_name} - Test MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")


# DATA -------------------------------------------------------------------------
# Create a DataFrame for the data
df = pd.read_csv('WECs_DataSet/' + filenames[1] + '.csv', header=None)

# X1, X2, ..., X16, Y1, Y2, ..., Y16, P1, P2, ..., P16, Powerall
df.columns = [f'X{i}' for i in range(1, X_grid_size)] +[f'Y{i}' for i in range(1, y_grid_size)]+ [f'P{i}' for i in range(1, power_output)] + ['Powerall']

### Process the data
# Define input features (X) and target variable (y)
X_set = df.iloc[:, :-1]
y_set = df['Powerall']

### Split data into training and testing sets
# Split the dataset into training and testing sets with a 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, test_size=0.2, random_state=42) 
# Determine the maximum target value in both the training and testing sets
max_target_value = max(np.max(y_train), np.max(y_test))
# Normalize the target values by dividing them by the maximum target value
# This scales the target values to the range [0, 1]
y_train = y_train / max_target_value
y_test = y_test / max_target_value

### Choose form of model:
# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Ridge Regression
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, y_train)

# Lasso Regression
lasso_reg = Lasso(alpha=1.0)
lasso_reg.fit(X_train, y_train)

# Decision Tree Regression
tree_reg = DecisionTreeRegressor(max_depth=10)
tree_reg.fit(X_train, y_train)

'''
# Random Forest Regression
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# Gradient Boosting Regression
gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_reg.fit(X_train, y_train)

# Support Vector Regression
svr_reg = SVR(kernel='linear', C=1.0)
svr_reg.fit(X_train, y_train)
'''

# K-Nearest Neighbors Regression
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train, y_train)

# MLP (Multi-layer Perceptron) Regression (Neural Network)
mlp_reg = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp_reg.fit(X_train, y_train)

# Make predictions and evaluate each model
models = [linear_reg, ridge_reg, lasso_reg, tree_reg, knn_reg, mlp_reg]
model_names = ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree", "K-Nearest Neighbors", "MLP"]
model_metrics = train_and_evaluate_individual_models(X_train, y_train, X_test, y_test, models, model_names)

dataprint.print_metrics(model_metrics)
    
# Perform hyperparameter tuning for each model
#hyperparameter_tuning_and_evaluation(X_train, y_train, X_test, y_test, models_and_params)


'''
cv_scores = cross_val_score(linear_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)
print("Cross-Validation RMSE Scores:")
print(cv_rmse_scores)

cv_scores = cross_val_score(linear_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)
print("Cross-Validation RMSE Scores:")
print(cv_rmse_scores)
'''