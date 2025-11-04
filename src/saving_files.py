from data_cleaning import data_clean
cleaned_data=data_clean()
cleaned_data.to_csv('/Users/raashidkhan/Desktop/GitHub-Projects/Machine-Learning-Project/data/cleaned/cleaned_data.csv',index=False)

from feature_engineering import feature_engineer
engineered_data=feature_engineer()
engineered_data.to_csv('/Users/raashidkhan/Desktop/GitHub-Projects/Machine-Learning-Project/data/engineered/engineered_data.csv',index=False)

from split import split
train_data,test_data=split()
train_data.to_csv('/Users/raashidkhan/Desktop/GitHub-Projects/Machine-Learning-Project/data/split/train_data.csv',index=False)
test_data.to_csv('/Users/raashidkhan/Desktop/GitHub-Projects/Machine-Learning-Project/data/split/test_data.csv',index=False)

from feature_selection import feature_select
selected_features, X, importance_df, features, max_features_,n_estimators_, model = feature_select()
import joblib
joblib.dump(model, '/Users/raashidkhan/Desktop/GitHub-Projects/Machine-Learning-Project/models/final_model.pkl')