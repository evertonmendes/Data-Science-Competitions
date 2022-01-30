import numpy as np
import pandas as pd


def drop_Nan_columns(df, drop_limit=0.7):
    ''' drop columns with my Nan values
    Args:
      df, DataFrame
      drop_limit, percentage limit of Nan values in a column
    Return:
      Nan_dropped, list of features dropped
    '''
    Nan_dropped = [feature for feature in df.columns if df[feature]
                   [df[feature].isna() == True].shape[0] > drop_limit*df.shape[0]]
    df.drop(columns=Nan_dropped, inplace=True)

    return Nan_dropped


def nunique_upperBound_columns(df, p_upper_bound=0.85, drop_nunique=False):
    '''get columns with unique values above upper bound
    Args:
      df, DataFrame
      upper_bound, percentage of samples above upper bound
      drop_nunique, drop the nunique_features above upper bound if True
    Return:
      nunique_features, dict of samples of above upper bound
    '''
    nunique_features = {feature: df[feature].nunique(
    ) for feature in df.columns if df[feature].nunique() > p_upper_bound*df.shape[0]}
    if drop_nunique:
        df.drop(columns=nunique_features.keys(), inplace=True)

    return nunique_features


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


def domain_reliability(df, columns, p_trustfull_above=0.01):
    '''creates a column with the domain reliability of each sample
    Args:
      df, DataFrame
      columns, list of columns to analyse realiability
      p_trustfull_above, percentage of trusted domains
    Return:
      domain_values, dict with the numbers of samples in each domain
    '''
    columns=[column for column in columns if column in df.columns]

    for column in columns:
        domain_values = df[column].value_counts()
        n_all_domains = domain_values.sum()
        trustfull_domains = domain_values[domain_values >
                                          p_trustfull_above*n_all_domains].to_dict()

        df[str(column)+'_reliability'] = [trustfull_domains[domain]/n_all_domains if domain in trustfull_domains else 0 for domain in df[column]]

    df.drop(columns=columns, inplace=True)

    return domain_values


def feature_eng(df, reliability_columns):
    '''applying feature engineering of the Dataframe
    Args:
      df, DataFrame
      reliability_columns, columns to be used in domain_reliability function
    Return:
      dropped_columns, columns dropped by drop_Nan_columns and nunique_upperBound_columns functions
    '''

    dropped_columns=(drop_Nan_columns(df, 0.9))
    for value in list(nunique_upperBound_columns(df, 0.85, True).keys()):
      if value not in dropped_columns:
        dropped_columns.append(value)
    domain_reliability(df, reliability_columns, p_trustfull_above=0.01)

    return dropped_columns
