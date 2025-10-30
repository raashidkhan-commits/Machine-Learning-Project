#So far we have chosen the model: RandomForest as it outperformed other models
#But we need to tune it as it shows underfitting

#So, we use GridSearchCV to perform the model on many hyperparameters
#And then check for the best parameters

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np

data=pd.read_csv('stratified_train_set.csv')
print (data.columns)
features=data.drop(columns='clean__median_house_value').columns
X=data[features]
y=data['clean__median_house_value']

model=RandomForestRegressor()
param_grid = [{'n_estimators': [100], 'max_features': [8]}]
           
grid_search=GridSearchCV(model, param_grid, cv=10,
                         scoring='neg_mean_squared_error',
                         return_train_score=True)
grid_search.fit(X,y)
#Best Parameters: {'max_features': 6, 'n_estimators': 30}

model.fit(X,y)

print (grid_search.best_estimator_)
print (grid_search.best_params_)
error=np.sqrt(-grid_search.best_score_)
print (error)

test_data=pd.read_csv('stratified_test_set.csv')
X_test=test_data[features]
y_test=test_data['clean__median_house_value']

test_pred=model.predict(X_test)
com_df_test=pd.DataFrame({'Actual Values:':y_test,'Predicted Values:':test_pred,'Error:':y_test-test_pred})
print (com_df_test)

from sklearn.metrics import mean_squared_error as mse
mse_=mse(y_test,test_pred)
final_rmse=np.sqrt(mse_)
print (final_rmse)