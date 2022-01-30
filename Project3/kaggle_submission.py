import joblib
import pandas as pd
from drive.MyDrive.Data_Science_Competitions.Project3.Utils.preprocessing.data_preprocessing import preproc_normalize, preprocess
import copy

def submission(dropped_columns, columns_to_keep):
    '''Read best model and predict the file Test of kaggle, at the end creates kaggle_submission.csv with predictions 
    Args:
      dropped_columns, columns to be dropped from Test file(obtained in training)
      columns_to_keep, columns with relevant information
    Return:
      None
    '''

    # load best model
    best_model = joblib.load('best_estimator.pkl')
    print(dropped_columns)

    # reading test file
    test_df = pd.read_csv('/content/drive/MyDrive/Data_Science_Competitions/Project3/Data/test.csv')

    submission_df = pd.DataFrame(data=[index+1 for index in test_df.index], columns=['ImageId'])

    #applying drop to the columns founded in training and reliability of some columns
    test_df.drop(columns=dropped_columns, inplace=True)


    # creating the features and targets for this study
    X = test_df[columns_to_keep]

    # getting the name of numerical and categorical columns
    numerical_features = X._get_numeric_data().columns.tolist()
    categorical_features = [attribute for attribute in X.columns.tolist() if attribute not in X._get_numeric_data().columns.tolist()]

    # applying the preprocessing using the scaler obtained from Training
    scaler = joblib.load('scaler.save')
    X_train, X_test, y_train, y_test, scaler = preprocess(X_train=X, categorical_features=categorical_features, numerical_features=numerical_features,
                                                          normalize_fn=preproc_normalize, scaler_fn=scaler, scaler_trigger=True, X_test_trigger=False)

    # prediction
    y_pred = best_model.predict(X_train)

    # kaggle submission
    submission_df['Label'] = y_pred
    submission_df.to_csv('kaggle_submission.csv', index=False)

