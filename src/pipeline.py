from data_cleaning import data_clean
from feature_engineering import feature_engineer
from split import split
from feature_selection import feature_select

def run_pipeline():
    print ('Cleaning data...')
    cleaned_data=data_clean()

    print ('Feature engineering...')
    engineered_data=feature_engineer()

    print ('Splitting data...')
    train_data,test_data=split()

    print ('Selecting features...')
    selected_features, X, importance_df, features, max_features_, n_estimators_, model = feature_select()

    print ('Pipeline Completed Successfully')

    return {
        "cleaned_data": cleaned_data,
        "engineered_data": engineered_data,
        "train_data": train_data,
        "test_data": test_data,
        "selected_features": selected_features,
        "importance_df": importance_df,
        "model": model}

if __name__ == "__main__":
    results = run_pipeline()
