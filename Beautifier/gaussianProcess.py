import sklearn.decomposition
import sklearn.gaussian_process
import scipy.optimize
import pickle
import os
import numpy as np

def fitPCA(trainX):
    pca = sklearn.decomposition.PCA(n_components=35)
    pca.fit(trainX)
    return pca

def trainGP(df, dstPath, trainPercentage=0.9, featureset="facefeatures"):
    #we need to train for both male and female
    grouped = df.groupby("gender")

    for gender, group in grouped:
        print("training %s GP" % gender)

        trainSize = int(group.shape[0] * trainPercentage)
        trainX = np.array(group[featureset][:trainSize].as_matrix().tolist())
        trainY = np.array(group["attractiveness"][:trainSize].as_matrix().tolist())

        testX = np.array(group[featureset][trainSize:].as_matrix().tolist())
        testY = np.array(group["attractiveness"][trainSize:].as_matrix().tolist())

        pca = fitPCA(trainX)
        # reducedTrainX = pca.transform(trainX)
        # reducedTestX = pca.transform(testX)


        bounds = np.zeros((2, 2))
        bounds[0, :] = [0.01, 0.1] #alpha bounds
        bounds[1, :] = [1.0, 3.0] #constant bounds

        def gpTraining(params):
            alpha = params[0]
            constant = params[1]

            kernel = sklearn.gaussian_process.kernels.ConstantKernel(constant,constant_value_bounds="fixed") * sklearn.gaussian_process.kernels.RBF(1.0, length_scale_bounds="fixed")
            gp = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10)

            gp.fit(trainX, trainY)

            score = gp.score(testX, testY)

            print("gp (%0.4f, %0.4f) = %0.10f" % (alpha, constant, score))
            return -score

        gpParameters = scipy.optimize.minimize(gpTraining, [0.05, 1.5], method='SLSQP', bounds=bounds, options={"maxiter": 5, "eps": 0.001})
        alpha, constant = gpParameters.x

        kernel = sklearn.gaussian_process.kernels.ConstantKernel(constant,constant_value_bounds="fixed") * sklearn.gaussian_process.kernels.RBF(1.0, length_scale_bounds="fixed")
        gp = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10)
        gp.fit(trainX, trainY)
        score = gp.score(testX, testY)
        print("gp (%0.4f, %0.4f) = %0.10f" % (alpha, constant, score))

        pickle.dump((pca,gp), open(os.path.join(dstPath,"GP_%s.p"%gender), "wb"))