import sklearn.linear_model
import sklearn.decomposition
import sklearn.model_selection
import sklearn.ensemble
import sklearn.pipeline
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import TransformerMixin
class CustomTransformer(TransformerMixin):
    def __init__(self, func):
        self.func = func

    def transform(self, X, *_):
        result = []
        for x in X:
            result.append(self.func(x))
        return result

    def fit(self, *_):
        return self

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = sklearn.model_selection.learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def fitPCA(trainX):
    pca = sklearn.decomposition.PCA(n_components=35)
    pca.fit(trainX)
    return pca

def trainGP(df, dstPath, featureset="facefeatures", train_on_PCA=True, generate_PCA=True, transformer_func=None):
    #we need to train for both male and female
    grouped = df.groupby("gender")

    for gender, group in grouped:
        print("training regression for {}'s on {} features".format(gender,featureset))

        X = np.array(group[featureset].as_matrix().tolist())
        Y = np.array(group["attractiveness"].as_matrix().tolist())

        if featureset == "facefeaturesCNN":
            X = X[:, 0:99]

        pipe = []

        if transformer_func == "facefeatures3D":
            pipe.append(('custom_transformer',CustomTransformer(transformer_func)))

        if generate_PCA or train_on_PCA:
            pca = fitPCA(X)
            if train_on_PCA:
                pipe.append(('pca',pca))
        else:
            pca = None

        #scale the data
        # pipe.append(('scaling',sklearn.preprocessing.StandardScaler()))

        estimator = sklearn.svm.SVR(kernel='rbf')
        # estimator = sklearn.linear_model.LinearRegression()
        # estimator = sklearn.ensemble.RandomForestRegressor()
        pipe.append(('estimator', estimator))

        pipeline = sklearn.pipeline.Pipeline(pipe)

        parameters_to_search = {'estimator__C': np.logspace(0, 2, 3), "estimator__epsilon":np.logspace(-2, 2, 5), "estimator__gamma": np.logspace(-2, 2, 5)}
        if train_on_PCA:
            parameters_to_search["pca__n_components"] = np.arange(30, 61, step=10)
        gridsearch = sklearn.model_selection.GridSearchCV(pipeline, parameters_to_search)
        gridsearch.fit(X,Y)

        print("Best parameters set found on development set:")
        print(gridsearch.best_params_)

        pipeline = gridsearch.best_estimator_


        score = sklearn.model_selection.cross_val_score(pipeline, X, Y).mean()
        print("Score with the entire dataset = %.2f" % score)

        plot_learning_curve(pipeline, "learning curve for linear regression", X, Y, train_sizes=np.linspace(.1, 1.0, 5))
        plt.show()

        pickle.dump((pca,pipeline), open(os.path.join(dstPath,"GP_%s.p"%gender), "wb"))