import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

pd.set_option('display.max_columns', None)  # Show all columns

data=pd.read_csv('machine_learning_data.csv')
print (data.head())
print (data.dtypes)
print ('Correlation Matrix:\n',data.drop(columns='ocean_proximity').corr())

#here also, we see that almost 2000 people belong to median_income of very poor (0-2 median),
#and almost 1000-1500 people belong to income category of very rich (6-14 median_income)
# and most importantly which means that almost 18000 people out of 21000 people belong 
# to an income_category of 2-6
data['median_income'].plot(kind='hist')
plt.xlabel('Median Income')  
plt.ylabel('Count')          
plt.title('Histogram of Median Income')
#plt.show() 


#Categorizing a continuos data
#This will help us in stratified sampling
#going to create 6 categories:
data['new_income_category']=pd.cut(data['median_income'],bins=[0,1.5,3,4.5,6,np.inf],labels=[1,2,3,4,5])
print (data['new_income_category'].value_counts())
print (data.head())
print (data.dtypes) #new_income_category is a category type, so let's convert to object
data['new_income_category']=data['new_income_category'].astype('Int64')
print (data.dtypes)

data['median_income'].hist()
data['new_income_category'].hist()
#plt.show()

numeric_cols=data.select_dtypes(include=['Int64','float64']).columns
object_cols=data.select_dtypes(include='object').columns
print ('Numeric Column Names:',numeric_cols,'\n','Object Column Names:',object_cols)

numeric_cols_pipeline=Pipeline([('imputer',SimpleImputer(strategy='median'),
                                 )])
object_cols_pipeline=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                               ('encoder',OneHotEncoder(handle_unknown='ignore'))])

full_pipeline_with_order=ColumnTransformer([('num_pipeline',numeric_cols_pipeline,numeric_cols),
                                            ('object_pipeline',object_cols_pipeline,object_cols)
                                            ])

cleaned_data_array=full_pipeline_with_order.fit_transform(data)
cleaned_data=pd.DataFrame(cleaned_data_array,columns=full_pipeline_with_order.get_feature_names_out())
print (cleaned_data)

cleaned_data.to_csv('cleaned_data.csv')