import sklearn.gaussian_process
import pickle
import os
import numpy as np

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

        bestScore = -100000
        bestGP = None
        kernel = sklearn.gaussian_process.kernels.ConstantKernel(1.4, constant_value_bounds="fixed") * sklearn.gaussian_process.kernels.RBF(1.0, length_scale_bounds="fixed")

        alpha = 0.0337
        gp = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10)
        gp.fit(trainX.copy(), trainY.copy())

        score = gp.score(testX,testY)
        if score > bestScore:
            bestScore = score
            bestGP = gp

        print("gp (%0.4f, %0.4f) = %0.10f"%(alpha,1.4,score))

        if bestGP is not None:
            pickle.dump(bestGP, open(os.path.join(dstPath,"GP_%s.p"%gender), "wb"))