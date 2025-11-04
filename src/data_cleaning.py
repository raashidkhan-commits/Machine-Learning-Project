import pandas as pd
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.pipeline import Pipeline
def data_clean():
    data=pd.read_csv('/Users/raashidkhan/Desktop/GitHub-Projects/Machine-Learning-Project/data/raw/machine_learning_data.csv')
    numeric_cols=data.drop(columns='median_house_value').select_dtypes(include=['float','int']).columns
    object_cols=data.select_dtypes(include='object').columns
    numeric_pipeline=Pipeline([('num',SimpleImputer(strategy='median'))])
    object_pipeline=Pipeline([('obj_imputer',SimpleImputer(strategy='most_frequent')),
                               ('obj_encoder',OneHotEncoder(sparse_output=False))])
    col_transformer=ColumnTransformer([('num',numeric_pipeline,numeric_cols),
                                       ('obj',object_pipeline,object_cols)
                                       ],verbose_feature_names_out=False)
    transformed_array=col_transformer.fit_transform(data)
    cleaned_data=pd.DataFrame(transformed_array,columns=col_transformer.get_feature_names_out())
    cleaned_data['median_house_value']=data['median_house_value']
    return cleaned_data