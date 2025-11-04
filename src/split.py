from sklearn.model_selection import StratifiedShuffleSplit
from feature_engineering import feature_engineer
import numpy as np
import pandas as pd

def split():
    data=feature_engineer()
    data['income_categories']=pd.cut(data['median_income'],bins=[0,1.5,3,4.5,6,np.inf],labels=[1,2,3,4,5])
    for train_ix, test_ix in StratifiedShuffleSplit(n_splits=1,random_state=42,test_size=0.2).split(data,data['income_categories']):
        train_data=data.iloc[train_ix].drop(columns='income_categories')
        test_data=data.iloc[test_ix].drop(columns='income_categories')
    return train_data,test_data