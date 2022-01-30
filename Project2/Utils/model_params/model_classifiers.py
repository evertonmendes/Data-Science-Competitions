from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier

from skopt.space import Real, Categorical, Integer


mlp_clf = {
    'model': Categorical([MLPClassifier()]),
    'model__activation': Categorical(['identity', 'logistic', 'tanh', 'relu']),
    'model__solver': Categorical(['sgd', 'adam']),
    'model__alpha': Real(0.0000001, 0.001,  'uniform'),
    'model__learning_rate': Categorical(['constant', 'invscaling', 'adaptive']),
    'model__learning_rate_init': Real(0.0000001, 0.001,  'uniform'),
    'model__power_t': Real(0.0005, 0.5, 'uniform'),
    'model__max_iter': Integer(100,5000, 'uniform'),
    'model__momentum': Real(0.1, 0.99, 'uniform'),
    'model__beta_1': Real(0.1, 0.99, 'uniform'),
    'model__beta_2': Real(0.1, 0.99, 'uniform'),
    'model__epsilon': Real(0.0000001, 0.001,  'uniform'),
}


bernoulli_clf={
    'model': Categorical([BernoulliNB()]),
    'model__alpha':Real(0.01, 0.99, 'uniform')
}

QDA_clf={
    'model':Categorical([QuadraticDiscriminantAnalysis()]),
    'model__tol':Real(0.0000001, 0.01,  'uniform')
}

PassiveAggressive_clf={
    'model': Categorical([PassiveAggressiveClassifier()]),
    'model__max_iter':Integer(1000, 10000, 'uniform'),
    'model__tol':Real(0.0000001, 0.01,  'uniform'),
    'model__C': Real(0.01, 0.99, 'uniform'),
    'model__loss': Categorical(['hinge', 'squared_hinge'])
}


ridge_clf_positive = {
    'model': Categorical([RidgeClassifier(positive=True)]),
    'model__tol': Real(0.0000001, 0.001,  'uniform'),
}

ridge_clf_false = {
    'model': Categorical([RidgeClassifier(positive=False)]),
    'model__solver': Categorical(['svd', 'cholesky', 'sparse_cg', 'sag', 'saga']),
    'model__tol': Real(0.0000001, 0.001,  'uniform')
}

perceptron_clf = {
    'model': Categorical([Perceptron(fit_intercept=False)]),
    'model__penalty': Categorical(['l2', 'l1', 'elasticnet']),
    'model__alpha': Real(0.00000001, 0.001, 'uniform'),
    'model__l1_ratio': Real(0.01, 0.99, 'uniform'),
    'model__max_iter': Integer(1000, 10000, 'uniform'),
    'model__tol': Real(0.0000001, 0.001,  'uniform'),
}


sgd_clf = {
    'model': Categorical([SGDClassifier(fit_intercept=False)]),
    'model__loss': Categorical(['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
    'model__alpha': Real(0.00000001, 0.001, 'uniform'),
    'model__max_iter': Integer(1000, 10000, 'uniform'),
    'model__epsilon': Real(0.0000001, 0.001,  'uniform'),
    'model__power_t': Real(0.01, 0.99, 'uniform'),
    'model__eta0': Real(0.01, 0.99, 'uniform'),
    'model__warm_start': Categorical([True, False]),
    'model__tol': Real(0.0000001, 0.001, 'uniform'),
    'model__penalty': Categorical(['l2', 'l1', 'elasticnet']),
    'model__l1_ratio': Real(0.01, 0.99, 'uniform'),
    'model__learning_rate': Categorical(['constant', 'optimal', 'invscaling', 'adaptive']),
}
