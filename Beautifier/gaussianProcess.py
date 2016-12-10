import sklearn.gaussian_process
import pickle
import os
import numpy as np

def trainGP(df, dstPath, trainPercentage=1.0):
    #we need to train for both male and female
    grouped = df.groupby("gender")

    for gender, group in grouped:
        print("training %s GP" % gender)

        trainSize = int(group.shape[0] * trainPercentage)
        trainX = group["facefeatures"][:trainSize].as_matrix()
        trainY = group["attractiveness"][:trainSize].as_matrix()

        trainX = np.array(trainX.tolist())
        trainX += np.random.normal(scale=0.00001, size=trainX.shape)
        trainY = np.array(trainY.tolist())
        # trainY += np.random.normal(scale=0.001, size=trainY.shape)

        gp = sklearn.gaussian_process.GaussianProcessRegressor(normalize_y=True)
        gp.fit(trainX, trainY)
        pickle.dump(gp, open(os.path.join(dstPath,"GP_%s.p"%gender), "wb"))