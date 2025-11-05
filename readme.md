This project is inspired by the California Housing case study from Hands-On Machine Learning with Scikit-Learn and TensorFlow by Aurélien Géron.
It walks through the complete machine learning process — starting from cleaning and preparing the data, creating new features, training a model, and finally evaluating its performance using cross-validation.



Key Notes:

* Cleaned raw housing data and handled missing values.
* Engineered new feature: `bedrooms_per_room = total_bedrooms / total_rooms`.
* Applied stratified sampling based on `median_income` for balanced train-test split.
* Used **RandomForestRegressor** as base model.
* Performed **GridSearchCV** to find best `n_estimators` and `max_features`.
* Identified most important features (kept those with importance > 0.02).
* Evaluated model using **10-fold cross-validation**.
                            Test RMSE: 47340.971787533206
                            Train RMSE: 18352.799359
                            Test R²: 0.8280409890014178
                            Train R²: 0.975
* Final test RMSE ≈ **47,300**, training RMSE ≈ **18,300**.
* Maintained clean modular structure.
* Create workflow pipeline.