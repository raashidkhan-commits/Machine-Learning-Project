import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error

pd.set_option('display.max_columns',None)

data=pd.read_csv('stratified_train_set.csv')
features=data.drop(columns='clean__median_house_value').columns
X=data[features]
y=data[['clean__median_house_value']]

regressor_model=DecisionTreeRegressor()
regressor_model.fit(X,y)
predictions=regressor_model.predict(X)
print (predictions)

comparison_df= (pd.DataFrame({'Actual House Values:':y.values.flatten(),
                     'Predicted House Values:':predictions.flatten()}))
print (comparison_df)
comparison_df['Error']=comparison_df['Predicted House Values:']-comparison_df['Actual House Values:']
print (comparison_df)

regressor_rmse=root_mean_squared_error(comparison_df['Predicted House Values:'],comparison_df['Actual House Values:'])
print (regressor_rmse)

#Before this project we had tried linear regression model and got a HIGH RMSE,meaning that the model
#was indicating underfitting
#Now, after using DecisionTreeRegressor, we are getting 0 RMSE error, which means model
#is overfit

#So how to prevent underfitting or overfitting?