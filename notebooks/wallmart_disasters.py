# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('dataxx.csv')
print(data.d.max())
# One-hot encode the categorical columns
encoder = OneHotEncoder(drop='first', sparse=False)
encoded_features = encoder.fit_transform(data[['store_id', 'item_id', 'month']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['store_id', 'item_id', 'month']))

# Concatenate the original data with the encoded features
data.reset_index(drop=True, inplace=True)
encoded_df.reset_index(drop=True, inplace=True)

# Concatenate the original data with the encoded features
data = pd.concat([data, encoded_df], axis=1)

# Drop original categorical columns & any other column not used
data = data.drop(columns=['store_id', 'state_id', 'cat_id', 'month', 'item_id', 'dept_id', 'event_name', 'event_type', 'id', 'wday', 'snap_CA', 'snap_TX', 'snap_WI', 'weekday'])
print(data)
print(data.isna().sum())


data['wm_yr_wk'] = data['wm_yr_wk'].astype(int)
print(data['wm_yr_wk'].dtype)

data.dtypes
print(data.date.min())
data.head()
print(data.date.max())
#################### LINEAR REGRESSION WITH EVENT/WEEKDAY BINARY ###############################
data = data.sort_values(by='date')
data.rename(columns={'profit': 'turnover'}, inplace=True)

data['lag_7'] = data['sell_price'].shift(7)
data['lag_14'] = data['sell_price'].shift(14)
data['lag_21'] = data['sell_price'].shift(21)
data.fillna(0, inplace=True)
training_data = data[(data['date'] >= '2011-01-29') & (data['date'] <= '2016-02-01')]

# Train-test split
X = training_data.drop(columns=['turnover', 'date', 'sales'])
print(X.columns)
y = training_data['sales']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)  # Splitting 60% training, 40% for validation and test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)  # Splitting the 40% into 20% validation and 20% test
print(X_train.head())
print(X_train.columns)

xgb_model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.225, max_depth=4)
lgb_model = LGBMRegressor(n_estimators=200, learning_rate=0.137, lambda_l1=0.5, lambda_l2=0.01, min_data_in_leaf=50, max_bin=255, num_leaves=63)
linear_model = LinearRegression()


xgb_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)
linear_model.fit(X_train, y_train)

# Predictions
xgb_val_predictions = xgb_model.predict(X_val) * X_val['sell_price']
lgb_val_predictions = lgb_model.predict(X_val) * X_val['sell_price']
linear_val_predictions = linear_model.predict(X_val) * X_val['sell_price']

xgb_test_predictions = xgb_model.predict(X_test) * X_test['sell_price']
lgb_test_predictions = lgb_model.predict(X_test) * X_test['sell_price']
linear_test_predictions = linear_model.predict(X_test) * X_test['sell_price']

# Evaluation
# Validation set
xgb_val_rmse = np.sqrt(mean_squared_error(y_val * X_val['sell_price'], xgb_val_predictions))
lgb_val_rmse = np.sqrt(mean_squared_error(y_val * X_val['sell_price'], lgb_val_predictions))
linear_val_rmse = np.sqrt(mean_squared_error(y_val * X_val['sell_price'], linear_val_predictions))

# Test set
xgb_test_rmse = np.sqrt(mean_squared_error(y_test * X_test['sell_price'], xgb_test_predictions))
lgb_test_rmse = np.sqrt(mean_squared_error(y_test * X_test['sell_price'], lgb_test_predictions))
linear_test_rmse = np.sqrt(mean_squared_error(y_test * X_test['sell_price'], linear_test_predictions))

# Additional metrics
xgb_r2 = r2_score(y_test, xgb_test_predictions)
lgb_r2 = r2_score(y_test, lgb_test_predictions)
linear_r2 = r2_score(y_test, linear_test_predictions)

xgb_mae = mean_absolute_error(y_test, xgb_test_predictions)
lgb_mae = mean_absolute_error(y_test, lgb_test_predictions)
linear_mae = mean_absolute_error(y_test, linear_test_predictions)
# print("Best Parameters for XGBoost: ", best_xgb_model)
# Best Parameters for XGBoost:  XGBRegressor(alpha=0.1, base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,
#              colsample_bytree=0.7, device=None, early_stopping_rounds=None,
#              enable_categorical=False, eval_metric=None, feature_types=None,
#              gamma=None, grow_policy=None, importance_type=None,
#              interaction_constraints=None, lambda=1,
#              learning_rate=0.1577777777777778, max_bin=None,
#              max_cat_threshold=None, max_cat_to_onehot=None,
#              max_delta_step=None, max_depth=7, max_leaves=None,
#              min_child_weight=1, missing=nan, monotone_constraints=None,
#              multi_strategy=None, n_estimators=100, n_jobs=None, ...)
# print("Best Parameters for LightGBM: ", best_lgb_model)
# Best Parameters for LightGBM:  LGBMRegressor(lambda_l1=0.5, lambda_l2=0.01, learning_rate=0.1366666666666667,
#               max_bin=255, min_data_in_leaf=50, n_estimators=200,
#               num_leaves=63)

print(f"XGBoost Validation RMSE: {xgb_val_rmse}, Test RMSE: {xgb_test_rmse}, MAE: {xgb_mae}")
# XGBoost Validation RMSE: 4.741505962963398, Test RMSE: 4.707142371165061, MAE: 1.9522095311763918

print(f"LightGBM Validation RMSE: {lgb_val_rmse}, Test RMSE: {lgb_test_rmse},  MAE: {lgb_mae}")
# LightGBM Validation RMSE: 4.635627810248236, Test RMSE: 4.605109430615783, MAE: 1.9483539024935013

print(f"Linear Regression Validation RMSE: {linear_val_rmse}, Test RMSE: {linear_test_rmse}, MAE: {linear_mae}")
# Linear Regression Validation RMSE: 5.208158783017889, Test RMSE: 5.1760292452986025, MAE: 2.006245420850325
def extract_ids(column_names, pattern):
    """Extract IDs from column names based on a given pattern."""
    ids = [re.search(pattern, col).group(1) for col in column_names if re.search(pattern, col)]
    return ids

# New Assignment: Sequential Predictions
prediction_start_date = pd.to_datetime('2016-02-02')
prediction_end_date = pd.to_datetime('2016-04-24')

data['date'] = pd.to_datetime(data['date'])
import re
def daily_predictions(start_date, end_date, model, retrain=False):

    current_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    predictions = []

    while current_date <= end_date:
        current_X = data[data['date'] >= current_date]

        if not current_X.empty:
            # Extracting store_id and item_id before dropping them
            store_ids = extract_ids(current_X.columns, r'store_id_(\d+)')
            item_ids = extract_ids(current_X.columns, r'item_id_(\d+)')

            # Preparing data for prediction
            prediction_X = current_X.drop(columns=['turnover', 'date', 'sales'])
            daily_prediction = model.predict(prediction_X) * prediction_X['sell_price']
            predictions.extend(daily_prediction)

            # Printing predictions with date, store_id, and item_id
            for store_id, item_id, prediction in zip(store_ids, item_ids, daily_prediction):
                print(f"Date: {current_date}, Store ID: {store_id}, Item ID: {item_id}, Prediction: {prediction}")

            # Optionally retrain model with new data
            if retrain:
                past_data = data[data['date'] <= current_date]
                X_new = past_data.drop(columns=['turnover', 'date', 'sales'])
                y_new = past_data['sales']
                model.fit(X_new, y_new)

        current_date += pd.Timedelta(days=1)
    return predictions

# Using the best model for daily predictions
xgb_daily_predictions = daily_predictions(prediction_start_date, prediction_end_date, lgb_model, retrain=True)
print(xgb_daily_predictions)

import matplotlib.pyplot as plt

feature_importances = xgb_model.feature_importances_

# Map feature importances to feature names
feature_names = X.columns
feature_importance_dict = {feature: importance for feature, importance in zip(feature_names, feature_importances)}

# Sort features by importance and get the top 20
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:35]

# Print the top 20 features
print("Top 20 Feature Importances:")
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")  # Adjusted to format with 4 decimal places for precision

# Plotting the top 20 feature importances
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_features)), [val[1] for val in sorted_features], align='center')
plt.yticks(range(len(sorted_features)), [val[0] for val in sorted_features])
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Top 20 Feature Importances in XGBoost')
plt.gca().invert_yaxis()  # To display the highest importance at the top
plt.show()

feature_importances = model_xgb.feature_importances_

# Map feature importances to feature names
feature_names = X.columns
feature_importance_dict = {feature: importance for feature, importance in zip(feature_names, feature_importances)}

# Sort features by importance
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Print or plot
print("Feature Importances:")
for feature, importance in sorted_features:
    print(f"{feature}: {importance}")

# Plotting feature importances
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_features)), [val[1] for val in sorted_features], align='center')
plt.yticks(range(len(sorted_features)), [val[0] for val in sorted_features])
plt.xlabel('Importance')
plt.title('Feature Importances in XGBoost')
plt.show()