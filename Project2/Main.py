from drive.MyDrive.Data_Science_Competitions.Project2.Train_Test_model import Train_model
from drive.MyDrive.Data_Science_Competitions.Project2.kaggle_submission import submission

def Main(list_of_calls, trigger_plot=False):
    '''Train models and create kaggle submission
    Args:
      list_of_calls, list with the number of calls for each model
      trigger_plot, if True, create plots of dependence, convergence, ROC and confusion Matrix
    Return:
      None
    '''
    
    classifiers = ['mlp_clf_n_calls', 'QDA_clf_n_calls', 'bernoulli_clf_n_calls', 'PassiveAggressive_clf_n_calls', 'sgd_n_calls', 'ridge_false_n_calls', 'ridge_positive_n_calls','perceptron_n_calls']
    #classifiers_calls=[30, 30, 40, 60, 30, 20]
    classifiers_calls = list_of_calls
    
    dict_of_calls = {classifiers[index]: classifiers_calls[index]
                     for index in range(len(classifiers_calls))}

    print(dict_of_calls)
    
    print("Training Model")
    dropped_columns=Train_model(**dict_of_calls, plot=trigger_plot)
    
    print("Creating kaggle submission")
    submission(dropped_columns)
