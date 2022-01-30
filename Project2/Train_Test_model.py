from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from drive.MyDrive.Data_Science_Competitions.Project2.Utils.preprocessing.data_treatment import undersample_boostrap, feature_eng
from drive.MyDrive.Data_Science_Competitions.Project2.Utils.preprocessing.data_preprocessing import preproc_normalize, preprocess
from skopt import BayesSearchCV
from skopt.plots import plot_objective
from drive.MyDrive.Data_Science_Competitions.Project2.Utils.model_params.model_classifiers import *
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import joblib
from skopt.plots import plot_objective
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence
from sklearn import metrics
import seaborn as sn


def Train_model(mlp_clf_n_calls, bernoulli_clf_n_calls, PassiveAggressive_clf_n_calls, sgd_n_calls, ridge_false_n_calls, ridge_positive_n_calls, perceptron_n_calls, QDA_clf_n_calls , plot=False):
    '''Apply feature engineering(data_treatment) and preprocessing to the train data. Furthermore, do optimization of hyperparameters by pipeline 
       and BayesSearch CV. At the end, save results and creat plots for analysis
    Args:
      model_n_calls,  number of calls for bayesian optimization
      plot, if plot is True, the function create plot of dependes, convergence, ROC and Confusion Matrix
    Return:
      dropped_columns, columns droppend from train data
    '''

    # reading the traininf file
    df_train_transaction = pd.read_csv('/content/drive/MyDrive/Data_Science_Competitions/Project2/Data/train_transaction.csv')
    df_train_identity = pd.read_csv('/content/drive/MyDrive/Data_Science_Competitions/Project2/Data/train_identity.csv')

    # concatening two datframe and aplying feature engeneering
    df_features = pd.concat([df_train_identity, df_train_transaction], axis=1)
    dropped_columns=feature_eng(df_features, ["P_emaildomain", "id_30", "id_31",
                "id_33", "id_34", "DeviceInfo", "card6", "R_emaildomain"])

    # creating the features and targets for this study
    X = df_features.drop(columns='isFraud')
    y = df_features['isFraud']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, shuffle=False)

    # undersample data(there is to many no Fraud in samples)
    X_train, y_train = undersample_boostrap(X_train, y_train, 10000)

    # preprocessing(replacing Nan values, Hot enconder and normalization)
    numerical_features = X._get_numeric_data().columns.tolist()
    categorical_features = [attribute for attribute in X.columns.tolist(
    ) if attribute not in X._get_numeric_data().columns.tolist()]
    X_train, X_test, y_train, y_test, scaler = preprocess(
        X_train, X_test, y_train, y_test, categorical_features, numerical_features, preproc_normalize)

    pipeline = Pipeline([
        ('model', MLPClassifier())
    ])

    optimizer = BayesSearchCV(estimator=pipeline, search_spaces=[(mlp_clf, mlp_clf_n_calls), (PassiveAggressive_clf, PassiveAggressive_clf_n_calls),(QDA_clf, QDA_clf_n_calls),(bernoulli_clf, bernoulli_clf_n_calls), (sgd_clf, sgd_n_calls),(ridge_clf_false, ridge_false_n_calls), (ridge_clf_positive, ridge_positive_n_calls), (perceptron_clf, perceptron_n_calls)], cv=3, scoring='roc_auc', verbose=2)
    ''' (mlp_clf, mlp_clf_n_calls),(PassiveAggressive_clf, PassiveAggressive_clf_n_calls),(QDA_clf, QDA_clf_n_calls),(bernoulli_clf, bernoulli_clf_n_calls), (sgd_clf, sgd_n_calls),(ridge_clf_false, ridge_false_n_calls), (ridge_clf_positive, ridge_positive_n_calls), (perceptron_clf, perceptron_n_calls)
     '''
    
    #fittimg models and cross-validation
    optimizer.fit(X_train, np.ravel(y_train))
    
    print("val. score: %s" % optimizer.best_score_)
    print("test score: %s" % optimizer.score(X_test, y_test))
    print("best params: %s" % str(optimizer.best_params_))
    
    # saving best model and results
    joblib.dump(optimizer.best_estimator_, 'best_estimator.pkl')
    np.save('my_results.npy', optimizer.cv_results_)


    #---IMAGES FOR ANALYSIS --- #

    classifiers = ['MLPClassifier', 'PassiveAggressiveClassifier', 'QuadraticDiscriminantAnalysis', 'BernoulliNB', 'SGDClassifier', 'RidgeClassifierFalse', 'RidgeClassifierPositive', 'Perceptron']
    #'MLPClassifier', 'PassiveAggressiveClassifier', 'QuadraticDiscriminantAnalysis', 'BernoulliNB', 'SGDClassifier', 'RidgeClassifierFalse', 'RidgeClassifierPositive', 'Perceptron', 'GaussianProcessClassifier'


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
        plt.savefig('Convegence.png')
        plt.clf()

        plt.title('ROC Best Score')
        plt.rcParams['figure.figsize'] = [15, 10]
        fig, ax = plt.subplots()
        y_pred = optimizer.best_estimator_.predict(X_test)
        ax.add_image = metrics.RocCurveDisplay.from_predictions(y_test, y_pred, ax=ax, name="Best estimator")
        plt.savefig("Best_estimator.png")
        plt.clf()

        plt.title("Confusion Matrix Best Score")
        metrics.plot_confusion_matrix(optimizer.best_estimator_, X_test, y_test, cmap="magma")
        plt.savefig("Confusion_Matrix.png")
        plt.clf()

        try:
          print(optimizer.best_estimator_.predict_proba(X_test))
        except:
          print("predict_proba does not exist for this classifier")


    # save scaler from of the training
    X = df_features.drop(columns='isFraud')
    y = df_features['isFraud']
    X_train, X_test, y_train, y_test, scaler = preprocess(
        X_train=X, y_train=y, categorical_features=categorical_features, numerical_features=numerical_features, normalize_fn=preproc_normalize, X_test_trigger=False)
    joblib.dump(scaler, 'scaler.save')

    return dropped_columns


