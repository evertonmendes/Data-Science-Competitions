import joblib
import pandas as pd
from drive.MyDrive.Data_Science_Competitions.Project2.Utils.preprocessing.data_preprocessing import preproc_normalize, preprocess
from drive.MyDrive.Data_Science_Competitions.Project2.Utils.preprocessing.data_treatment import domain_reliability
import copy

def submission(dropped_columns):
    '''Read best model and predict the file Test of kaggle, at the end creates kaggle_submission.csv with predictions 
    Args:
      dropped_columns, columns to be dropped from Test file(obtained in training)
    Return:
      None
    '''

    # load best model
    best_model = joblib.load('best_estimator.pkl')
    print(dropped_columns)

    # reading test file
    df_test_transaction = pd.read_csv('/content/drive/MyDrive/Data_Science_Competitions/Project2/Data/test_transaction.csv')
    df_test_identity = pd.read_csv('/content/drive/MyDrive/Data_Science_Competitions/Project2/Data/test_identity.csv')
    df_features = pd.concat([df_test_identity, df_test_transaction], axis=1)

    submission_data=[int(value) for value in df_features['TransactionID'].values[:, 1]]
    submission_df = pd.DataFrame(data=submission_data, columns=['TransactionID'])

    #applying drop to the columns founded in training and reliability of some columns
    print(list(df_features.columns))
    dropped_columns=[value.replace("_", "-") for value in dropped_columns]
    df_features.drop(columns=dropped_columns, inplace=True)

    domain_reliability(df_features, ["P_emaildomain", "id-30", "id-31", "id-33", "id-34", "DeviceInfo", "card6", "R_emaildomain"])


    # applying feature engeneering
    #feature_eng(df_features, ["P_emaildomain", "id_30", "id_31",
    #            "id_33", "id_34", "DeviceInfo", "card6", "R_emaildomain"])

    # creating the features and targets for this study
    X = df_features

    # getting the name of numerical and categorical columns
    numerical_features = X._get_numeric_data().columns.tolist()
    categorical_features = [attribute for attribute in X.columns.tolist(
    ) if attribute not in X._get_numeric_data().columns.tolist()]

    # applying the preprocessing using the scaler obtained from Training
    scaler = joblib.load('scaler.save')
    X_train, X_test, y_train, y_test, scaler = preprocess(X_train=X, categorical_features=categorical_features, numerical_features=numerical_features,
                                                          normalize_fn=preproc_normalize, scaler_fn=scaler, scaler_trigger=True, X_test_trigger=False)

    # prediction
    y_pred = best_model.predict(X_train)

    # kaggle submission
    submission_df['isFraud'] = y_pred
    submission_df.to_csv('kaggle_submission.csv', index=False)

