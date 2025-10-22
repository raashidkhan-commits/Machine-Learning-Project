import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

pd.set_option('display.max_columns', None)  # Show all columns
data=pd.DataFrame(pd.read_csv('Cleaned Data.csv'))
print (data)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["clean__new_income_category"]):
    strat_train_set = data.loc[train_index]
    strat_test_set=data.loc[test_index]
print ('Stratified Train Set:',strat_train_set,'\n','Stratified Test Set',strat_test_set)
print ((strat_train_set['clean__new_income_category'].value_counts())/strat_test_set['clean__new_income_category'].value_counts())
#Here we see that the ratio as per categories is almost same, i.e 4
#we have successfully split the dataframe using StratifiedShuffleSplit

#The clean__new_income_category was used only to split the df as per this category
#Now, we dont need it, so let's drop it
for set_ in (strat_train_set, strat_test_set):
    set_.drop("clean__new_income_category", axis=1, inplace=True)

strat_train_set.to_csv('Stratified Train Set.csv')
strat_test_set.to_csv('Stratified Test Set.csv')

#Loaded  cleaned dataset
#Used StratifiedShuffleSplit based on clean__new_income_category
#Verified the ratio consistency
#Dropped the helper column (since it was only needed for stratification)
#Saved  train and test sets separately as csv files




