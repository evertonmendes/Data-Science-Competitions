from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def preproc_normalize(X_train=None, X_test=None, y_train=None, y_test=None, scaler=None, scaler_trigger=False, X_test_trigger=True):
    '''normilize the data with MinMaxScaler
    Args:
      X_train, X_test, y_train, y_test
    Return:
      X_train, X_test, y_train, y_test
    '''

    if scaler_trigger == False:
        scaler = MinMaxScaler()
        scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    if X_test_trigger == True:
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def preprocess(X_train=None, X_test=None, y_train=None, y_test=None, categorical_features=None, numerical_features=None, normalize_fn=None, scaler_fn=None, scaler_trigger=False, X_test_trigger=True):
    '''replace Nan values of categorical and numerical features. Moreover, transform categorical features in numerical data
    Args:
      X_train, X_test, y_train, y_test
      categorical_features, list with the name of the categorical columns
      numerical_features, list with the name of the numerical columns
    Return:
      X_train, X_test, y_train, y_test
    '''

    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder())])

    transformation = ColumnTransformer(
        transformers=[
            ('numerical transformation', numerical_pipeline, numerical_features),
            ('categorical transformation',
             categorical_pipeline, categorical_features),
        ])

    X_train = transformation.fit_transform(X_train)
    if X_test_trigger == True:
        X_test = transformation.transform(X_test)

    if scaler_trigger == False and X_test_trigger == False:
        X_train, X_test, y_train, y_test, scaler = normalize_fn(
            X_train=X_train, y_train=y_train, X_test_trigger=False)

    elif scaler_trigger == True and X_test_trigger == False:
        X_train, X_test, y_train, y_test, scaler = normalize_fn(
            X_train=X_train, y_train=y_train, X_test_trigger=False, scaler=scaler_fn, scaler_trigger=True)

    elif scaler_trigger == False and X_test_trigger == True:
        X_train, X_test, y_train, y_test, scaler = normalize_fn(
            X_train, X_test, y_train, y_test)
    else:
        X_train, X_test, y_train, y_test, scaler = normalize_fn(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, scaler=scaler_fn, X_test_trigger=False, scaler_trigger=True)

    return X_train, X_test, y_train, y_test, scaler
