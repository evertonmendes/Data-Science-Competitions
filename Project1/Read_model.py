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
import matplotlib.pyplot as plt


house_df=pd.read_csv('train.csv')


house_df['AgeSold']      = house_df['YrSold'] - house_df['YearBuilt']
house_df['AgeRemodSold'] = house_df['YrSold'] - house_df['YearRemodAdd']
house_df['GarageAgeBlt'] = house_df['YrSold'] - house_df['GarageYrBlt']


target_col  = 'SalePrice'

future_cols = [
    'MoSold',
    'YrSold',
    'SaleType',
    'SaleCondition',
    target_col
]

drop_fe_cols = [
    'YearBuilt',
    'YearRemodAdd',
    'GarageYrBlt'
]

y = house_df['SalePrice']

# remover colunas futuras do X
X = house_df.drop(future_cols, axis=1)

# remove vars do feature eng
X.drop(drop_fe_cols, axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=0.65,
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
print(tfidf_params)

gscv=pipeline.set_params(**tfidf_params)



gscv.fit(X_train, y_train)

pred=gscv.predict(X_test)

#plt.plot(y_test, label="real")
#plt.plot(pred, label="prediction")
#plt.legend()
#plt.show()
mse=mean_squared_error(y_test, pred)
mape=mean_absolute_percentage_error(y_test, pred)

print('Best mse: {}\n Best mape: {}\n'.format(mse, mape))

