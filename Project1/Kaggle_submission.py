from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.base import BaseEstimator
from RegSwitcher import RegSwitcher
import joblib
from feature_eng import feature_eng

house_df=pd.read_csv('train.csv')

house_df, X, y=feature_eng(house_df, y_activation=True)


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=0.8,
    random_state=42
)

len(y_train), len(y_test)



pipeline_num = Pipeline(
    steps=[
        ('imputer', SimpleImputer()),
        ('scaler' , MinMaxScaler())
    ]
)

ct = ColumnTransformer([
    (
        'num_transf', 
        pipeline_num, 
        make_column_selector(dtype_exclude=object)
    ),
    (
        'categ_transf', 
        OneHotEncoder(sparse=False, handle_unknown='ignore'), 
        make_column_selector(dtype_include=object)
    )
])

pipeline = Pipeline(
    steps=[
        ('ct', ct),
        ('reg', RegSwitcher())
    ]
)


# Load best parameters
tfidf_params = joblib.load('best_tfidf.pkl')


gscv=pipeline.set_params(**tfidf_params)
gscv.fit(X, y)


#gscv.fit(X_train, y_train)

test_df=pd.read_csv('test.csv')
test_df, test_X, test_y=feature_eng(test_df, y_activation=False)

pred=gscv.predict(test_X)
#mse=mean_squared_error(y_test, pred)
#mape=mean_absolute_percentage_error(y_test, pred)


submission_df=pd.DataFrame(test_df['Id'].values, columns=['Id'])
submission_df['SalePrice']=pred

submission_df.to_csv('kaggle_submission.csv', index=False)


#print('Best mse: {}\n Best mape: {}\n'.format(mse, mape))
