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
import joblib as jb
from feature_eng import feature_eng
from sklearn import svm
from sklearn.linear_model import SGDRegressor
from sklearn.naive_bayes import GaussianNB
#from skopt import skopt


house_df=pd.read_csv('train.csv')
#print(house_df)


house_df, X, y=feature_eng(house_df, y_activation=True)

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

parameters = [
    {

        'ct__num_transf__imputer__strategy': ['mean', 'median'],
        'ct__num_transf__scaler'           : [MinMaxScaler(), StandardScaler()],


        'reg__estimator': [XGBRegressor()],
        'reg__estimator__eta':[index/10.00 for index in range(1, 3)],
      	'reg__estimator__gamma':[2**index for index in range(1, 4)],
	'reg__estimator__min_child_weight': [2**index for index in range(1, 4)],
	'reg__estimator__objective':['reg:squarederror', 'reg:squaredlogerror', 'reg:pseudohubererror'],
        'reg__estimator__rate_drop': [index/10.00 for index in range(1, 3)],
        'reg__estimator__skip_drop': [index/10.00 for index in range(1, 3)],
    },
    {

        'ct__num_transf__imputer__strategy': ['mean', 'median'],
        'ct__num_transf__scaler'           : [MinMaxScaler(), StandardScaler()],


        'reg__estimator': [svm.SVR()],
        'reg__estimator__kernel':['linear','poly', 'rbf', 'sigmoid'],
	'reg__estimator__epsilon':[1.0/(10.0**index) for index in range(1, 4)],
	
    },
    {

        'ct__num_transf__imputer__strategy': ['mean', 'median'],
        'ct__num_transf__scaler'           : [MinMaxScaler(), StandardScaler()],


        'reg__estimator': [SGDRegressor()],
	'reg__estimator__loss':['huber', 'epsilon_insensitive','squared_epsilon_insensitive'],
        'reg__estimator__penalty':['l2', 'l1', 'elasticnet'],
	'reg__estimator__alpha':[1.0/(10.0**index) for index in range(3, 6)],
	'reg__estimator__eta0':[1.0/(10.0**index) for index in range(1, 3)],
	
    },
    {

        'ct__num_transf__imputer__strategy': ['mean', 'median'],
        'ct__num_transf__scaler'           : [MinMaxScaler(), StandardScaler()],


        'reg__estimator': [GaussianNB()],
	
    }
	
]


gscv =GridSearchCV(pipeline, parameters, verbose=2)
# cv=5, n_jobs=5, return_train_score=False,

gscv.fit(X_train, y_train)

pred=gscv.best_estimator_.predict(X_test)
mse=mean_squared_error(y_test, pred)
mape=mean_absolute_percentage_error(y_test, pred)

print('Best model:\n', gscv.best_params_)
print('Best mse: {}\n Best mape: {}\n'.format(mse, mape))

#joblib.dump(grid.best_estimator_, 'best_tfidf.pkl', compress = 1) # this unfortunately includes the LogReg

jb.dump(gscv.best_params_, 'best_tfidf.pkl', compress = 1) #



''',
    {
        'ct__num_transf__imputer__strategy': ['mean', 'median'],
        'ct__num_transf__scaler'           : [MinMaxScaler(), StandardScaler()],
        
        'reg__estimator': [DecisionTreeRegressor()],
        'reg__estimator__criterion' : ["mse", "friedman_mse", "mae", "poisson"],
        'reg__estimator__splitter' : ["best", "random"],
        'reg__estimator__min_samples_split' : [2**index for index in range(1, 3)],
        'reg__estimator__max_features' : ["auto", "sqrt", "log2"],

        'reg__estimator__max_depth' : [2**index for index in range(1, 3)],

            

    },
    {
        'ct__num_transf__imputer__strategy': ['mean', 'median'],
        'ct__num_transf__scaler'           : [MinMaxScaler(), StandardScaler()],

        'reg__estimator': [RandomForestRegressor()],
        'reg__estimator__n_estimators' : [2**index for index in range(1, 3)],
        'reg__estimator__criterion' : ["mse", "mae", "mape"],
        'reg__estimator__min_samples_split' : [2**index for index in range(1, 3)],
        'reg__estimator__max_features' : ["auto", "sqrt", "log2"],
        'reg__estimator__max_depth' : [2**index for index in range(1, 3)]
    }
'''

