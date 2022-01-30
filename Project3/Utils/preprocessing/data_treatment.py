import numpy as np
import pandas as pd
from drive.MyDrive.Data_Science_Competitions.Project3.Utils.Data_analysis.analysis import drop_no_information

def drop_Nan_columns(df, drop_limit=0.7):
    ''' drop columns with my Nan values
    Args:
      df, DataFrame
      drop_limit, percentage limit of Nan values in a column
    Return:
      df, new DataFrame with dropped columns
      Nan_dropped, list of features dropped
    '''
    Nan_dropped = [feature for feature in df.columns if df[feature]
                   [df[feature].isna() == True].shape[0] > drop_limit*df.shape[0]]
    df.drop(columns=Nan_dropped, inplace=True)

    return df, Nan_dropped


def undersample_boostrap(inputs: pd.DataFrame, targets: pd.DataFrame, bootstrap_size=1000):
    '''undersample a Daframe with with features(inputs) and targets
    Args:
      inputs, DataFrame with the features
      targets, Datframe with the targets
    Return:
      undersampled_data,
      undersampled_targets,
    '''
    min_sample = min(targets.value_counts().tolist())

    undersampled_data = pd.DataFrame(columns=inputs.columns)
    undersampled_targets = pd.DataFrame()

    # undersample
    for class_type in targets.value_counts().index:

        indices_class = np.where(targets == class_type)[0]

        indices_atrr_sample = np.random.choice(
            a=indices_class, size=min_sample, replace=False)
        undersampled_data = undersampled_data.append(
            inputs.iloc[indices_atrr_sample])
        undersampled_targets = undersampled_targets.append(
            targets.iloc[indices_atrr_sample].tolist())

    # bootstrap
    for class_type in targets.value_counts().index:

        indices_class = np.where(targets == class_type)[0]

        indices_atrr_sample = np.random.choice(
            a=indices_class, size=bootstrap_size, replace=True)
        undersampled_data = undersampled_data.append(
            inputs.iloc[indices_atrr_sample])
        undersampled_targets = undersampled_targets.append(
            targets.iloc[indices_atrr_sample].tolist())

    return undersampled_data, undersampled_targets


def feature_eng(df):
    '''applying feature engineering of the Dataframe
    Args:
      df, DataFrame
    Return:
      dropped_columns, columns dropped by drop_Nan_columns and nunique_upperBound_columns functions
      columns_to_keep, columns with relevant information
    '''

    new_df, dropped_columns=(drop_Nan_columns(df, 0.9))
    new_df, columns_to_keep=drop_no_information(new_df)
    
    return new_df, dropped_columns, columns_to_keep
