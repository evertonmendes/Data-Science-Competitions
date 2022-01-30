from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from drive.MyDrive.Data_Science_Competitions.Project3.Utils.preprocessing.data_treatment import undersample_boostrap, feature_eng
from drive.MyDrive.Data_Science_Competitions.Project3.Utils.preprocessing.data_preprocessing import preproc_normalize, preprocess
from skopt import BayesSearchCV
from skopt.plots import plot_objective
from drive.MyDrive.Data_Science_Competitions.Project3.Utils.model_params.model_classifiers import *
from sklearn.pipeline import Pipeline
import joblib
from skopt.plots import plot_objective
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence
from sklearn import metrics
import seaborn as sn
from sklearn.svm import SVC


def Train_model(lgbm_clf_n_calls, xgbrf_dart_clf_n_calls, xgbrf_gbtree_clf_n_calls, xgb_gbtree_clf_n_calls, xgb_dart_clf_n_calls, gaussianNB_clf_n_calls, Decision_Tree_clf_n_calls, Random_Forest_clf_n_calls, AdaBoost_clf_n_calls, gradientBooster_clf_n_calls, HistGradientBooster_clf_n_calls, knc_clf_n_calls, Knn_clf_n_calls, svc_clf_n_calls, PassiveAggressive_clf_n_calls, sgd_n_calls, ridge_false_n_calls, ridge_positive_n_calls, perceptron_n_calls, QDA_clf_n_calls , plot=False):
    '''Apply feature engineering(data_treatment) and preprocessing to the train data. Furthermore, do optimization of hyperparameters by pipeline 
       and BayesSearch CV. At the end, save results and creat plots for analysis
    Args:
      model_n_calls,  number of calls for bayesian optimization
      plot, if plot is True, the function create plot of dependes, convergence, ROC and Confusion Matrix
    Return:
      dropped_columns, columns dropped from train data
      columns_to_keep, columns with relevant information
    '''

    ''' '''
    # reading the training file
    train_df = pd.read_csv('/content/drive/MyDrive/Data_Science_Competitions/Project3/Data/train.csv')

    # concatening two datframe and aplying feature engeneering
    new_train_df, dropped_columns, columns_to_keep=feature_eng(train_df)

    # creating the features and targets for this study
    X = new_train_df
    y = train_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=False)

    # undersample data(there is to many no Fraud in samples)
    X_train, y_train = undersample_boostrap(X_train, y_train, 0)

    # preprocessing(replacing Nan values, Hot enconder and normalization)
    numerical_features = X._get_numeric_data().columns.tolist()
    categorical_features = [attribute for attribute in X.columns.tolist() if attribute not in X._get_numeric_data().columns.tolist()]
    X_train, X_test, y_train, y_test, scaler = preprocess(X_train, X_test, y_train, y_test, categorical_features, numerical_features, preproc_normalize)

    #pipeline for many models
    pipeline = Pipeline([
        ('model', SVC())
    ])

    #models for pipeline
    Search_spaces=[
        (lgbm_clf, lgbm_clf_n_calls),
        (gaussianNB_clf, gaussianNB_clf_n_calls),
        (Decision_Tree_clf, Decision_Tree_clf_n_calls),
        (Random_Forest_clf, Random_Forest_clf_n_calls),
        (AdaBoost_clf, AdaBoost_clf_n_calls),
        (knc_clf, knc_clf_n_calls),
        (svc_clf, svc_clf_n_calls),
        (PassiveAggressive_clf,PassiveAggressive_clf_n_calls),
        (ridge_clf_false, ridge_false_n_calls),
        (ridge_clf_positive, ridge_positive_n_calls),
        (perceptron_clf, perceptron_n_calls,),
       
    ]

    '''
    (xgbrf_dart_clf, xgbrf_dart_clf_n_calls),
    (xgb_dart_clf, xgb_dart_clf_n_calls),
    (HistGradientBooster_clf, HistGradientBooster_clf_n_calls),
    (gradientBooster_clf, gradientBooster_clf_n_calls),
    (xgbrf_gbtree_clf, xgbrf_gbtree_clf_n_calls),
    (xgb_gbtree_clf, xgb_gbtree_clf_n_calls),
    (Knn_clf, Knn_clf_n_calls),
    (sgd_clf, sgd_n_calls),
     (QDA_clf, QDA_clf_n_calls)
        '''

    optimizer = BayesSearchCV(estimator=pipeline, search_spaces=Search_spaces, cv=3, scoring='accuracy', verbose=3)
    
    #fittimg models and cross-validation
    optimizer.fit(X_train, np.ravel(y_train))
    
    print("val. score: %s" % optimizer.best_score_)
    print("test score: %s" % optimizer.score(X_test, y_test))
    print("best params: %s" % str(optimizer.best_params_))
    
    # saving best model and results
    joblib.dump(optimizer.best_estimator_, 'best_estimator.pkl')
    np.save('my_results.npy', optimizer.cv_results_)


    #---IMAGES FOR ANALYSIS --- #

    classifiers = [
        'LGBMClassifier',
        'GaussianNB',
        'DecisionTreeClassifier',
        'RandomForestClassifier',
        'AdaBoostClassifier',
        'NearestCentroid',
        'SVC',
        'PassiveAggressiveClassifier',
        'RidgeClassifierFalse',
        'RidgeClassifierPositive',
        'Perceptron',
        
        ]

    ''' 
    'XGBRFClassifier_dart',
    'XGBClassifier_dart',
    'HistGradientBoostingClassifier',
    'GradientBoostingClassifier',
       'XGBRFClassifier_gbtree',
        'XGBClassifier_gbtree',
           'KNeighborsClassifier',
           'SGDClassifier',
           'QuadraticDiscriminantAnalysis'
        ''' 

    if plot == True:

        for i in range(len(optimizer.optimizer_results_)):
            plt.title(classifiers[i])
            _ = plot_objective(optimizer.optimizer_results_[i])
            plt.savefig(classifiers[i]+'_dependence.png')
            plt.clf()

        plt.rcParams['figure.figsize'] = [15, 15]
        plt.title("Convergence of models")
        clf_plot = ((classifiers[index], optimizer.optimizer_results_[
                    index]) for index in range(len(classifiers)))
        plot = plot_convergence(*clf_plot)
        plot.legend(loc="best", prop={'size': 6}, numpoints=1)
        plt.savefig('Convergence.png')
        plt.clf()


        plt.title("Confusion Matrix Best Score")
        metrics.plot_confusion_matrix(optimizer.best_estimator_, X_test, y_test, cmap="inferno")
        plt.savefig("Confusion_Matrix.png")
        plt.clf()

        try:
          print(optimizer.best_estimator_.predict_proba(X_test))
        except:
          print("predict_proba does not exist for this classifier")


    # save scaler from of the training
    X = new_train_df
    y = train_df['label']
    X_train, X_test, y_train, y_test, scaler = preprocess(
        X_train=X, y_train=y, categorical_features=categorical_features, numerical_features=numerical_features, normalize_fn=preproc_normalize, X_test_trigger=False)
    joblib.dump(scaler, 'scaler.save')

    return dropped_columns, columns_to_keep


