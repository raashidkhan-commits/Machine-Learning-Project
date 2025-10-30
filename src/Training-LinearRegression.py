import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

pd.set_option('display.max_columns', None)  # Show all columns


train_data=pd.DataFrame(pd.read_csv('stratified_train_set.csv'))
print (train_data)

features=(train_data.drop(columns='clean__median_house_value')).columns
X=train_data[features]
y=train_data[['clean__median_house_value']]
print ('X(Features):\n',X,'\n','y(Target):\n',y)

model=LinearRegression()
model.fit(X,y)
predictions=model.predict(X)
print (predictions)
comparison_df= (pd.DataFrame({'Actual House Values:':y.values.flatten(),
                     'Predicted House Values:':predictions.flatten()}))
comparison_df['Error']=comparison_df['Predicted House Values:']- comparison_df['Actual House Values:']
print (comparison_df)
lin_rmse=root_mean_squared_error(comparison_df['Predicted House Values:'],comparison_df['Actual House Values:'])
print ('RMSE:',lin_rmse)
# Here, a High RMSE of 69000 dollars on training data is a sign of underfitting.
#High RSME means underfitting
#Low RMSE on training set, but high RMSE on test set is a sign of overfitting.
#Rule of Thumb
        #High RMSE on both training and test sets → underfitting
        #Low RMSE on training but high on test → overfitting



from sklearn.metrics import r2_score
r2 = r2_score(y, predictions)
print("Model Performance (%):", r2*100)

#saving model:
import joblib
joblib.dump(model, "LinearRegression Model.pkl")
# and later...
my_model_loaded = joblib.load("LinearRegression Model.pkl")