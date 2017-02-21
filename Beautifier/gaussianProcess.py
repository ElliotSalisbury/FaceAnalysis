import sklearn.decomposition
import sklearn.gaussian_process
import pickle
import os
import numpy as np

def fitPCA(trainX):
    pca = sklearn.decomposition.PCA(n_components=35)
    pca.fit(trainX)
    return pca

def trainGP(df, dstPath, trainPercentage=0.9, featureset="facefeatures", train_on_PCA=True, generate_PCA=True):
    #we need to train for both male and female
    grouped = df.groupby("gender")

    for gender, group in grouped:
        print("training %s GP" % gender)

        trainSize = int(group.shape[0] * trainPercentage)
        trainX = np.array(group[featureset][:trainSize].as_matrix().tolist())
        trainY = np.array(group["attractiveness"][:trainSize].as_matrix().tolist())

        testX = np.array(group[featureset][trainSize:].as_matrix().tolist())
        testY = np.array(group["attractiveness"][trainSize:].as_matrix().tolist())

        if featureset == "facefeaturesCNN":
            trainX = trainX[:, 0:99]
            testX = testX[:, 0:99]

        if generate_PCA:
            pca = fitPCA(trainX)
            if train_on_PCA:
                trainX = pca.transform(trainX)
                testX = pca.transform(testX)
        else:
            pca = None

        svm = sklearn.svm.SVR()
        svm.fit(trainX, trainY)

        score = svm.score(testX, testY)
        print("svm = %0.10f" % (score))

        pickle.dump((pca,svm), open(os.path.join(dstPath,"GP_%s.p"%gender), "wb"))