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
#train_score=cross_val_score(model,X_train,y_train,scoring='neg_mean_squared_error',cv=10)
#def get_score(score):
    #positive_=np.sqrt(-score)
    #print ('Mean:',positive_.mean())
    #print ('Standard Deviation:',positive_.std())

#get_score(train_score)
#score on validation set:
        # Mean: 49569.81734327374
        #Standard Deviation: 2031.6818309754183

#Now, let us find scores on training set:
model.fit(X,y)
predictions=model.predict(X)
print (predictions)
print (y)
print (predictions-y)

print("RMSE:",root_mean_squared_error(predictions,y))
#here training error:
#RMSE: 18475.65467383569

#AT LAST WE FOUND THAT THE VALIDATION ERROR IS STILL VERY VERY LARGER COMPARED
#TO TRAINING ERROR
#EVEN THOUGH THE RANDOMFOREST MODEL IS THE BEST PERMORMING OF ALL, BUT STILL
#THE MODEL IS OVERFITTING