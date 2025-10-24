import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

data=pd.DataFrame(pd.read_csv('stratified_train_set.csv'))
print (data.columns)

features=data.drop(columns='clean__median_house_value').columns
X=data[features]
y=data[['clean__median_house_value']]

model_lin=LinearRegression()

#for linear model:
scores=cross_val_score(model_lin,X,y,scoring='neg_mean_squared_error',cv=10)
print (scores)
lin_rmse_scores=np.sqrt(-scores)
print (lin_rmse_scores)

def all_scores(scores):
    print ('Scores:',scores)
    print ('Mean:',scores.mean())
    print ('Standard Deviation:',scores.std())

all_scores(lin_rmse_scores)

#for decision tree model:
model_regressor=DecisionTreeRegressor()
scores=cross_val_score(model_regressor,X,y,scoring='neg_mean_squared_error',cv=10)
print (scores)
reg_rmse_scores=np.sqrt(-scores)
all_scores(reg_rmse_scores)
# as we can see, the regressor model performs worse than the linear regressor model

model_random=RandomForestRegressor()
scores=cross_val_score(model_random,X,y,scoring='neg_mean_squared_error',cv=10)
random_rmse_scores=np.sqrt(-scores)
all_scores(random_rmse_scores)
#here we can see that the random forest outperforms all models