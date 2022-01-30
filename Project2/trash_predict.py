from sklearn import metrics
import matplotlib.pyplot as plt
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
import os

def models_prediction():
  '''predict the train data for three models: random_model, only_Fraud_model and only_no_Fraud_model. At the end, creates a ROC Plot for the models
  Args:
    None
  Return:
    None
  '''

  def random_model(X_test):
      rd_list = [rd.random() for i in range(len(X_test))]
      predicted = [1 if value > 0.5 else 0 for value in rd_list]
      return predicted


  def Fraud_model(X_test):
      predicted = [1 for i in range(len(X_test))]
      return predicted


  def not_Fraud_model(X_test):

      predicted = [0 for i in range(len(X_test))]
      return predicted


  # reading data
  df_train_transaction = pd.read_csv('/content/drive/MyDrive/Data_Science_Competitions/Project2/Data/train_transaction.csv')
  df_train_identity = pd.read_csv('/content/drive/MyDrive/Data_Science_Competitions/Project2/Data/train_identity.csv')

  # concatening two datframe
  df_features = pd.concat([df_train_identity, df_train_transaction], axis=1)

  #features and targets
  X = df_features.drop(columns='isFraud')
  y = df_features['isFraud']

  pred_random = random_model(X)
  pred_fraud = Fraud_model(X)
  pred_no_fraud = not_Fraud_model(X)

  predictions = {'random_model': pred_random,
                'fraud_model': pred_fraud, 'no_fraud_model': pred_no_fraud}

  plt.rcParams['figure.figsize'] = [15, 10]
  fig, ax = plt.subplots()
  ax.set_title("ROC of 3 models")
  for model_name in predictions.keys():
      ax.add_image = metrics.RocCurveDisplay.from_predictions(y, predictions[model_name], ax=ax, name=model_name)

  plt.savefig('trash_models.png')


'''
fpr, tpr, _ = metrics.roc_curve(y_test, prediction)
roc_auc = metrics.auc(fpr, tpr)

'''
'''
plt.rcParams['figure.figsize'] = [15, 10]
fig, ax = plt.subplots()


ax.add_image = metrics.RocCurveDisplay.from_predictions(y_test, predicted, ax=ax, name=classificador_name)


plt.show()

from scipy.stats import ttest_ind

t_statistic, p_values=ttest_ind(best_classifiers["Multi-Layer Perceptron (15,)"]["scores"], best_classifiers["KNN k=5"]["scores"])

'''
