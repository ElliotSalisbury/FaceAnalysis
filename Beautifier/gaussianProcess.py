import sklearn.decomposition
import sklearn.gaussian_process
import pickle
import os
import numpy as np

def fitPCA(trainX):
    pca = sklearn.decomposition.PCA(n_components=35)
    pca.fit(trainX)
    return pca

def trainGP(df, dstPath, trainPercentage=0.9):
    #we need to train for both male and female
    grouped = df.groupby("gender")

    for gender, group in grouped:
        print("training %s GP" % gender)

        trainSize = int(group.shape[0] * trainPercentage)
        trainX = np.array(group["facefeatures"][:trainSize].as_matrix().tolist())
        trainY = np.array(group["attractiveness"][:trainSize].as_matrix().tolist())

        testX = np.array(group["facefeatures"][trainSize:].as_matrix().tolist())
        testY = np.array(group["attractiveness"][trainSize:].as_matrix().tolist())

        pca = fitPCA(trainX)
        reducedTrainX = pca.transform(trainX)
        reducedTestX = pca.transform(testX)

        bestScore = -100000
        bestGP = None
        for alpha in np.linspace(0.01,0.1, 20):
            for constant in np.linspace(1.0, 3.0, 30):
                kernel = sklearn.gaussian_process.kernels.ConstantKernel(constant, constant_value_bounds="fixed") * sklearn.gaussian_process.kernels.RBF(1.0, length_scale_bounds="fixed")

                gp = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10)
                gp.fit(reducedTrainX, trainY)

                score = gp.score(reducedTestX,testY)
                if score > bestScore:
                    bestScore = score
                    bestGP = gp

                print("gp (%0.4f, %0.4f) = %0.10f"%(alpha,constant,score))

        if bestGP is not None:
            pickle.dump((pca,bestGP), open(os.path.join(dstPath,"GP_%s.p"%gender), "wb"))