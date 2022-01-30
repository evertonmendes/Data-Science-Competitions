from skopt.space import Real, Categorical, Integer

from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier


lgbm_clf={
    'model':Categorical([LGBMClassifier()]),
    'model__boosting_type':Categorical(['gbdt', 'dart', 'goss']),
    'model__num_leaves':Integer(8, 128, 'uniform'),
    'model__max_depth':Integer(4, 128, 'uniform'),
    'model__learning_rate':Real(0.001, 0.4, 'uniform'),
    'model__n_estimators':Integer(100, 150, 'uniform'),
    'model__subsample_for_bin':Integer(200000, 250000, 'uniform'),
    'model__min_split_gain':Real(0, 0.1, 'uniform'),
    'model__min_child_weight':Real(0.000001, 0.001,  'uniform'),
    'model__min_child_samples':Integer(12, 40, 'uniform'),
    'model__subsample':Real(0.8, 1, 'uniform'),
    'model__colsample_bytree':Real(0.8, 1, 'uniform'),
}





gaussianNB_clf={
    'model':Categorical([GaussianNB()]),
    'model__var_smoothing':Real(0.0000000001, 0.0000001, 'uniform')
}

Decision_Tree_clf={
    'model':Categorical([DecisionTreeClassifier()]),
    'model__criterion':Categorical(['gini', 'entropy']),
    'model__splitter':Categorical(['best', 'random']),
    'model__min_samples_split':Integer(2, 8, 'uniform'),
    'model__max_features':Categorical(['sqrt', 'log2']),
    'model__max_depth':Integer(4, 128, 'uniform'),
    'model__min_samples_leaf':Integer(1,4,'uniform'),
}

Random_Forest_clf={
    'model':Categorical([RandomForestClassifier()]),
    'model__n_estimators':Integer(100, 120, 'uniform'),
    'model__criterion':Categorical(['gini', 'entropy']),
    'model__max_depth':Integer(4, 128, 'uniform'),
    'model__min_samples_split':Integer(2, 8, 'uniform'),
    'model__min_samples_leaf':Integer(1,4,'uniform'),
    'model__max_features':Categorical(['sqrt', 'log2']),
    'model__max_samples':Real(0.01, 0.99, 'uniform')
}
AdaBoost_clf={
    'model':Categorical([AdaBoostClassifier()]),
    'model__n_estimators':Integer(40, 100, 'uniform'),
    'model__learning_rate':Real(0.8, 1.2, 'uniform'),
    'model__algorithm':Categorical(['SAMME', 'SAMME.R']),
}

gradientBooster_clf={
    'model':Categorical([GradientBoostingClassifier()]),
    'model__loss':Categorical(['deviance']),
    'model__learning_rate':Real(0.0001, 0.1, 'uniform'),
    'model__n_estimators':Integer(80, 100, 'uniform'),
    'model__subsample':Real(0.7, 1, 'uniform'),
    'model__criterion':Categorical(['friedman_mse', 'squared_error']),
    'model__min_samples_split':Integer(2, 8, 'uniform'),
    'model__min_samples_leaf':Integer(1,4,'uniform'),
    'model__max_depth':Integer(2, 64, 'uniform'),
    'model__max_features':Categorical(['sqrt', 'log2']),
    'model__tol':Real(0.000001, 0.01,  'uniform')
}

HistGradientBooster_clf={
    'model':Categorical([HistGradientBoostingClassifier()]),
    'model__loss':Categorical(['categorical_crossentropy']),
    'model__learning_rate':Real(0.0001, 0.1, 'uniform'),
    'model__max_iter':Integer(500, 2000, 'uniform'),
    'model__max_leaf_nodes':Integer(20, 50, 'uniform'),
    'model__max_depth':Integer(2, 64, 'uniform'),
    'model__min_samples_leaf':Integer(1,20,'uniform'),    
    'model__tol':Real(0.00000001, 0.0001,  'uniform'),
}

knc_clf={
    'model':Categorical([NearestCentroid()]),
    'model__metric':Categorical(['euclidean', 'manhattan', 'chebyshev', 'minkowski'])
}


#'ball_tree',

svc_clf={
    'model':Categorical([SVC()]),
    'model__kernel':Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
    'model__degree':Integer(2, 10, 'uniform'),
    'model__gamma':Categorical(['scale', 'auto']),
    'model__tol':Real(0.0000001, 0.01,  'uniform'),
}

QDA_clf={
    'model':Categorical([QuadraticDiscriminantAnalysis()]),
    'model__tol':Real(0.0000001, 0.01,  'uniform')
}

PassiveAggressive_clf={
    'model': Categorical([PassiveAggressiveClassifier()]),
    'model__max_iter':Integer(1000, 10000, 'uniform'),
    'model__tol':Real(0.0000001, 0.01,  'uniform'),
    'model__C': Real(0.01, 0.99, 'uniform'),
    'model__loss': Categorical(['hinge', 'squared_hinge'])
}


ridge_clf_positive = {
    'model': Categorical([RidgeClassifier(positive=True)]),
    'model__tol': Real(0.0000001, 0.001,  'uniform'),
    'model__alpha':Real(0.00001, 1,  'uniform'),
    'model__max_iter':Integer(1000, 10000, 'uniform')
}

ridge_clf_false = {
    'model': Categorical([RidgeClassifier(positive=False)]),
    'model__solver': Categorical(['svd', 'cholesky', 'sparse_cg', 'sag', 'saga']),
    'model__tol': Real(0.0000001, 0.001,  'uniform'),
    'model__alpha':Real(0.00001, 1,  'uniform'),
    'model__max_iter':Integer(1000, 10000, 'uniform')
}

perceptron_clf = {
    'model': Categorical([Perceptron( fit_intercept=False)]),
    'model__penalty': Categorical(['l2', 'l1', 'elasticnet']),
    'model__alpha': Real(0.00000001, 0.001, 'uniform'),
    'model__l1_ratio': Real(0.01, 0.99, 'uniform'),
    'model__max_iter': Integer(1000, 10000, 'uniform'),
    'model__tol': Real(0.0000001, 0.001,  'uniform'),
}


sgd_clf = {
    'model': Categorical([SGDClassifier(fit_intercept=False)]),
    'model__loss': Categorical(['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
    'model__alpha': Real(0.00000001, 0.001, 'uniform'),
    'model__max_iter': Integer(1000, 10000, 'uniform'),
    'model__epsilon': Real(0.0000001, 0.001,  'uniform'),
    'model__power_t': Real(0.01, 0.99, 'uniform'),
    'model__eta0': Real(0.01, 0.99, 'uniform'),
    'model__warm_start': Categorical([True, False]),
    'model__tol': Real(0.0000001, 0.001, 'uniform'),
    'model__penalty': Categorical(['l2', 'l1', 'elasticnet']),
    'model__l1_ratio': Real(0.01, 0.99, 'uniform'),
    'model__learning_rate': Categorical(['constant', 'optimal', 'invscaling', 'adaptive']),
}




'''

Decision_Tree_reg={
    'model':Categorical([DecisionTreeRegressor()]),
    'model_criterion':Categorical(['squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
    'model_splitter':Categorical(['best', 'random']),
    'model_min_samples_split':Integer(2, 8, 'uniform'),
    'model_max_features':Categorical(['sqrt', 'log2']),
    'model_max_depth':Integer(4, 128, 'uniform'),
    'model_min_samples_leaf':Integer(1,4,'uniform'),
}

Knn_clf={
    'model':Categorical([KNeighborsClassifier()]),
    'model__weights':Categorical(['uniform', 'distance']),
    'model__algorithm':Categorical([ 'kd_tree', 'brute']),
    'model__leaf_size':Integer(10, 50, 'uniform'),
    'model__p':Integer(1, 4, 'uniform'),
    'model__metric':Categorical(['euclidean', 'manhattan', 'chebyshev', 'minkowski']),
    'model__n_neighbors':Integer(3, 6, 'uniform')
}

Random_Forest_reg={
    'model':Categorical([RandomForestRegressor()]),
    'model_n_estimators':Integer(100, 300, 'uniform'),
    'model_criterion':Categorical(['squared_error', 'absolute_error', 'poisson']),
    'model_max_depth':Integer(4, 128, 'uniform'),
    'model_min_samples_split':Integer(2, 8, 'uniform'),
    'model_min_samples_leaf':Integer(1,4,'uniform'),
    'model_max_features':Categorical(['sqrt', 'log2']),
    'model_max_samples':Real(0.01, 0.99, 'uniform')
}


AdaBoost_reg={
    'model':Categorical([AdaBoostRegressor()]),
    'model_base_estimator_criterion':Categorical(['squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
    'model_base_estimator_splitter':Categorical(['best', 'random']),
    'model_base_estimator_min_samples_split':Integer(2, 8, 'uniform'),
    'model_base_estimator_max_features':Categorical(['sqrt', 'log2']),
    'model_base_estimator_max_depth':Integer(4, 128, 'uniform'),
    'model_n_estimators':Integer(40, 70, 'uniform'),
    'model_learning_rate':Real(1, 3, 'uniform'),
    'model_loss':Categorical(['linear', 'square', 'exponential'])
}


GradientBoosting_reg={
    'model':Categorical([GradientBoostingRegressor()]),
    'model_loss':Categorical(['squared_error', 'absolute_error', 'huber', 'quantile']),
    'model_learning_rate':Real(0.001, 0.4, 'uniform'),
    'model_n_estimators':Integer(100, 250, 'uniform'),
    'model_criterion':Categorical(['friedman_mse', 'squared_error', 'mse', 'mae']),
    'model_min_samples_split':Integer(2, 8, 'uniform'),
    'model_min_samples_leaf':Integer(1,4,'uniform'),
    'model_max_depth':Integer(4, 128, 'uniform'),
    'model_max_features':Categorical(['sqrt', 'log2']),
    'model_alpha':Real(0.1, 0.9, 'uniform'),
    'model_tol':Real(0.0000001, 0.001,  'uniform')
}


xgb_gbtree_reg={
    'model':Categorical([XGBRegressor()]),
    'model_booster':Categorical(['gbtree']),
    'model_eta':Real(0.01, 0.99, 'uniform'),
    'model_gamma':Integer(0, 4, 'uniform'),
    'model_max_depth':Integer(4, 128, 'uniform'),
    'model_min_child_weight':Integer(0, 4, 'uniform'),
    'model_max_delta_step':Integer(0, 7, 'uniform'),
    'model_subsample':Real(0.01, 0.99, 'uniform'),
    'model_colsample_bytree':Real(0.01, 0.99, 'uniform'),
    'model_colsample_bylevel':Real(0.01, 0.99, 'uniform'),
    'model_colsample_bynode':Real(0.01, 0.99, 'uniform'),
    'model_n_estimators':Integer(100, 300, 'uniform'),
    'model_objective':Categorical(['reg:squarederror', 'reg:squaredlogerror', 'reg:logistic', 'reg:pseudohubererror']),
    'model_eval_metric':Categorical(['rmse', 'rmsle', 'mae', 'mape', 'mphe'])
}



xgb_dart_reg={
    'model':Categorical([XGBRegressor()]),
    'model_booster':Categorical([ 'dart']),
    'model_eta':Real(0.01, 0.99, 'uniform'),
    'model_gamma':Integer(0, 4, 'uniform'),
    'model_max_depth':Integer(4, 128, 'uniform'),
    'model_min_child_weight':Integer(0, 4, 'uniform'),
    'model_max_delta_step':Integer(0, 7, 'uniform'),
    'model_subsample':Real(0.01, 0.99, 'uniform'),
    'model_colsample_bytree':Real(0.01, 0.99, 'uniform'),
    'model_colsample_bylevel':Real(0.01, 0.99, 'uniform'),
    'model_colsample_bynode':Real(0.01, 0.99, 'uniform'),
    'model_n_estimators':Integer(100, 300, 'uniform'),
    'model_sample_type':Categorical(['uniform', 'weighted']),
    'model_normalize_type':Categorical(['tree', 'forest']),
    'model_rate_drop':Real(0.01, 0.99, 'uniform'),
    'model_skip_drop':Real(0.01, 0.99, 'uniform'),
    'model_objective':Categorical(['reg:squarederror', 'reg:squaredlogerror', 'reg:logistic', 'reg:pseudohubererror']),
    'model_eval_metric':Categorical(['rmse', 'rmsle', 'mae', 'mape', 'mphe'])
}

xgb_linear_reg={
    'model':Categorical([XGBRegressor()]),
    'model_booster':Categorical(['gblinear']),
    'model_feature_selector':Categorical(['cyclic', 'shuffle', 'random', 'greedy', 'thrifty', ]),
    'model_updater':Categorical(['shotgun', 'coord_descent']),
    'model_objective':Categorical(['reg:squarederror', 'reg:squaredlogerror', 'reg:logistic', 'reg:pseudohubererror']),
    'model_eval_metric':Categorical(['rmse', 'rmsle', 'mae', 'mape', 'mphe'])
}


lgbm_reg={
    'model':Categorical([LGBMRegressor()]),
    'model_boosting_type':Categorical(['gbdt', 'dart', 'goss', 'rf']),
    'model_num_leaves':Integer(8, 128, 'uniform'),
    'model_max_depth':Integer(4, 128, 'uniform'),
    'model_learning_rate':Real(0.001, 0.4, 'uniform'),
    'model_n_estimators':Integer(100, 300, 'uniform'),
    'model_subsample_for_bin':Integer(200000, 250000, 'uniform'),
    'model_min_split_gain':Real(0, 0.1, 'uniform'),
    'model_min_child_weight':Real(0.001, 0.000001, 'uniform'),
    'model_min_child_samples':Integer(12, 40, 'uniform'),
    'model_subsample':Real(0.8, 1, 'uniform'),
    'model_colsample_bytree':Real(0.8, 1, 'uniform'),

}

xgbrf_dart_clf={
    'model':Categorical([XGBRFClassifier()]),
    'model__booster':Categorical([ 'dart']),
    'model__eta':Real(0.01, 0.99, 'uniform'),
    'model__gamma':Integer(0, 4, 'uniform'),
    'model__max_depth':Integer(4, 128, 'uniform'),
    'model__min_child_weight':Integer(0, 4, 'uniform'),
    'model__max_delta_step':Integer(0, 7, 'uniform'),
    'model__subsample':Real(0.01, 0.99, 'uniform'),
    'model__colsample_bytree':Real(0.01, 0.99, 'uniform'),
    'model__colsample_bylevel':Real(0.01, 0.99, 'uniform'),
    'model__colsample_bynode':Real(0.01, 0.99, 'uniform'),
    'model__n_estimators':Integer(100, 120, 'uniform'),
    'model__sample_type':Categorical(['uniform', 'weighted']),
    'model__normalize_type':Categorical(['tree', 'forest']),
    'model__rate_drop':Real(0.01, 0.99, 'uniform'),
    'model__skip_drop':Real(0.01, 0.99, 'uniform'),
    'model__objective':Categorical(['reg:squarederror', 'reg:squaredlogerror', 'reg:logistic', 'reg:pseudohubererror']),
}


xgb_dart_clf={
    'model':Categorical([XGBClassifier()]),
    'model__booster':Categorical([ 'dart']),
    'model__eta':Real(0.01, 0.99, 'uniform'),
    'model__gamma':Integer(0, 4, 'uniform'),
    'model__max_depth':Integer(4, 128, 'uniform'),
    'model__min_child_weight':Integer(0, 4, 'uniform'),
    'model__max_delta_step':Integer(0, 7, 'uniform'),
    'model__subsample':Real(0.01, 0.99, 'uniform'),
    'model__colsample_bytree':Real(0.01, 0.99, 'uniform'),
    'model__colsample_bylevel':Real(0.01, 0.99, 'uniform'),
    'model__colsample_bynode':Real(0.01, 0.99, 'uniform'),
    'model__n_estimators':Integer(100, 120, 'uniform'),
    'model__sample_type':Categorical(['uniform', 'weighted']),
    'model__normalize_type':Categorical(['tree', 'forest']),
    'model__rate_drop':Real(0.01, 0.99, 'uniform'),
    'model__skip_drop':Real(0.01, 0.99, 'uniform'),
    'model__objective':Categorical(['reg:squarederror', 'reg:squaredlogerror', 'reg:logistic', 'reg:pseudohubererror']),
}


xgbrf_gbtree_clf={
    'model':Categorical([XGBRFClassifier()]),
    'model__booster':Categorical(['gbtree']),
    'model__eta':Real(0.01, 0.99, 'uniform'),
    'model__gamma':Integer(0, 4, 'uniform'),
    'model__max_depth':Integer(4, 128, 'uniform'),
    'model__min_child_weight':Integer(0, 4, 'uniform'),
    'model__max_delta_step':Integer(0, 7, 'uniform'),
    'model__subsample':Real(0.01, 0.99, 'uniform'),
    'model__colsample_bytree':Real(0.01, 0.99, 'uniform'),
    'model__colsample_bylevel':Real(0.01, 0.99, 'uniform'),
    'model__colsample_bynode':Real(0.01, 0.99, 'uniform'),
    'model__n_estimators':Integer(100, 120, 'uniform'),
    'model__objective':Categorical(['reg:squarederror', 'reg:squaredlogerror', 'reg:logistic', 'reg:pseudohubererror']),
}






xgb_gbtree_clf={
    'model':Categorical([XGBClassifier()]),
    'model__booster':Categorical(['gbtree']),
    'model__eta':Real(0.01, 0.99, 'uniform'),
    'model__gamma':Integer(0, 4, 'uniform'),
    'model__max_depth':Integer(4, 128, 'uniform'),
    'model__min_child_weight':Integer(0, 4, 'uniform'),
    'model__max_delta_step':Integer(0, 7, 'uniform'),
    'model__subsample':Real(0.01, 0.99, 'uniform'),
    'model__colsample_bytree':Real(0.01, 0.99, 'uniform'),
    'model__colsample_bylevel':Real(0.01, 0.99, 'uniform'),
    'model__colsample_bynode':Real(0.01, 0.99, 'uniform'),
    'model__n_estimators':Integer(100, 120, 'uniform'),
    'model__objective':Categorical(['reg:squarederror', 'reg:squaredlogerror', 'reg:logistic', 'reg:pseudohubererror']),
}


'''


