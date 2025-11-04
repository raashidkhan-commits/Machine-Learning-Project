from data_cleaning import data_clean
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

def feature_engineer():
    data=data_clean()
    data['bedrooms_per_room']=data['total_rooms']/data['total_bedrooms']
    return data