from Neighbor import NeighborComparative


def feature_eng(df, y_activation=False):
    '''feature engineering of House Prices - Advanced Regression Techniques kaggle competition
    Args:
        df, kaggle dataframe
        y_activation, create y if y_activation is True else return y=None
    Returns:
        df kaggle dataframe
        X(DataFrame) ,features of analysis
        y(DataFrame), target of analysis
    '''

    df['AgeSold'] = df['YrSold'] - df['YearBuilt']
    df['AgeRemodSold'] = df['YrSold'] - df['YearRemodAdd']
    df['GarageAgeBlt'] = df['YrSold'] - df['GarageYrBlt']

    if y_activation == True:
        target_col = 'SalePrice'
        future_cols = [
            'MoSold',
            'YrSold',
            'SaleType',
            'SaleCondition',
            target_col
        ]
    else:
        future_cols = [
            'MoSold',
            'YrSold',
            'SaleType',
            'SaleCondition',
        ]

    drop_fe_cols = [
        'YearBuilt',
        'YearRemodAdd',
        'GarageYrBlt'
    ]

    X = df.drop(future_cols, axis=1)
    X.drop(drop_fe_cols, axis=1, inplace=True)

    if y_activation == True:
        y = df['SalePrice']
    else:
        y = None

    X = NeighborComparative(X)

    return df, X, y
