import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np

data=pd.read_csv('stratified_train_set.csv')
#Feature Engineering Training Set
data["pop_per_household"] = data["clean__population"] / data["clean__households"]
data["rooms_per_household"] = data["clean__total_rooms"] / data["clean__households"]
data["bedrooms_per_room"] = data["clean__total_bedrooms"] / data["clean__total_rooms"] 


features=data.drop(columns='clean__median_house_value').columns
X=data[features]
y=data['clean__median_house_value']

model=RandomForestRegressor()
param_grid = [{'n_estimators': [100], 'max_features': [8,10]}]
grid_search = GridSearchCV(
    model, 
    param_grid, 
    cv=10,
    scoring='neg_mean_squared_error',
    return_train_score=True
    )
grid_search.fit(X,y)
feature_importances=grid_search.best_estimator_.feature_importances_
sorted_features=sorted(zip(feature_importances,features),reverse=True)
importance_df=pd.DataFrame(sorted_features,columns=['Importance', 'Feature Name'])
print (importance_df)


#Dropping least important features from train set:
data=data.drop(columns=['cleaned__ocean_proximity_<1H OCEAN','cleaned__ocean_proximity_NEAR OCEAN',
                                          'cleaned__ocean_proximity_NEAR BAY',
                                          'cleaned__ocean_proximity_ISLAND'])
data.to_csv('Feature_Engineered_Training_Set.csv',index=False)


#Feature Engineering Test Set, Dropping least important features:
data2=pd.read_csv('stratified_test_set.csv')
data2["pop_per_household"] = data2["clean__population"] / data2["clean__households"]
data2["rooms_per_household"] = data2["clean__total_rooms"] / data2["clean__households"]
data2["bedrooms_per_room"] = data2["clean__total_bedrooms"] / data2["clean__total_rooms"] 
data2=data2.drop(columns=['cleaned__ocean_proximity_<1H OCEAN','cleaned__ocean_proximity_NEAR OCEAN',
                                          'cleaned__ocean_proximity_NEAR BAY',
                                          'cleaned__ocean_proximity_ISLAND'])
data2.to_csv('Feature_Engineered_Test_Set.csv',index=False)

