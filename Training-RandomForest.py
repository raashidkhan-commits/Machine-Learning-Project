import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error

train_data=pd.read_csv('stratified_train_set.csv')
test_data=pd.read_csv('stratified_test_set.csv')
features=train_data.drop(columns='clean__median_house_value').columns

X=train_data[features]
y=train_data['clean__median_house_value']

model=RandomForestRegressor()
model.fit(X,y)
predictions=model.predict(X)

comparison_df=pd.DataFrame({'Actual House Price':y,'Predicted House Price':predictions,'Error':predictions-y})
print (comparison_df)

print("RMSE:",root_mean_squared_error(predictions,y))
#here training error:
#RMSE: 18475.65467383569


#from sklearn.metrics import r2_score
#r2 = r2_score(y, predictions)
#print("Model Performance (%):", r2*100)

#saving model:
#import joblib
#joblib.dump(model, "RandomForest Model.pkl")
# and later...
#my_model_loaded = joblib.load("RandomForest Model.pkl")
#print (my_model_loaded)