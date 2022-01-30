from drive.MyDrive.Data_Science_Competitions.Project3.Train_Test_model_Scikit import Train_model
from drive.MyDrive.Data_Science_Competitions.Project3.kaggle_submission import submission

def Main(dict_of_calls, trigger_plot=False):
    '''Train models and create kaggle submission
    Args:
      list_of_calls, list with the number of calls for each model
      trigger_plot, if True, create plots of dependence, convergence, ROC and confusion Matrix
    Return:
      None
    '''
    
    '''
    classifiers = [
        'lgbm_clf_n_calls',
        'xgbrf_dart_clf_n_calls',
        'xgbrf_gbtree_clf_n_calls',
        'xgb_gbtree_clf_n_calls',
        'xgb_dart_clf_n_calls',
        'gaussianNB_clf_n_calls',
        'Decision_Tree_clf_n_calls',
        'Random_Forest_clf_n_calls',
        'AdaBoost_clf_n_calls',
        'gradientBooster_clf_n_calls',
        'HistGradientBooster_clf_n_calls',
        'knc_clf_n_calls',
        'Knn_clf_n_calls',
        'svc_clf_n_calls',
        'PassiveAggressive_clf_n_calls',
        'sgd_n_calls',
        'ridge_false_n_calls',
        'ridge_positive_n_calls',
        'perceptron_n_calls',
        'QDA_clf_n_calls']

    '''
    #classifiers_calls=[30, 30, 40, 60, 30, 20]
    

    print(dict_of_calls)
    
    print("Training Model")
    dropped_columns, columns_to_keep=Train_model(**dict_of_calls, plot=trigger_plot)
    
    print("Creating kaggle submission")
    submission(dropped_columns, columns_to_keep)
