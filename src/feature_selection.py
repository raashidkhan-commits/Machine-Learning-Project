from split import split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd

def feature_select():
    def best_parameters():
        data = split()[0]
        features = data.drop(columns='median_house_value').columns
        X = data[features]
        y = data['median_house_value']
        model1 = RandomForestRegressor(random_state=42)
        param_grid = {'n_estimators': [100], 'max_features': [8]}
        search = GridSearchCV(model1,param_grid,cv=10,scoring='neg_mean_squared_error',verbose=2,return_train_score=True)
        search.fit(X, y)
        best = search.best_params_
        max_features_, n_estimators_ = best['max_features'], best['n_estimators']
        return max_features_, n_estimators_, X, y, features
    max_features_, n_estimators_, X, y, features = best_parameters()

    def train_using_best_parameters():
        model = RandomForestRegressor(random_state=42,max_features=max_features_,n_estimators=n_estimators_)
        model.fit(X, y)
        feature_importances=model.feature_importances_
        x = pd.DataFrame({'Feature Importance': feature_importances,'Feature Name': features})
        importance_df = x.sort_values(by='Feature Importance', ascending=False)
        selection_ = int((importance_df['Feature Importance'] > 0.02).sum())
        selected_features = list(importance_df[:selection_]['Feature Name'])
        return selected_features, importance_df, model
    selected_features,importance_df, model=train_using_best_parameters()
    return selected_features, X, importance_df, features, max_features_,n_estimators_, model