import numpy as np
import cv2
import pickle
import os
import scipy
from sklearn import gaussian_process
from Beautifier.warpFace import warpFace
from US10K.US10K import loadUS10KFacialFeatures, loadUS10KPCAGP
from RateMe.RateMe import loadRateMeFacialFeatures, loadRateMePCAGP
from Beautifier.faceFeatures import getNormalizingFactor, getFaceFeatures

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

scriptFolder = os.path.dirname(os.path.realpath(__file__))
faceLines = np.load(os.path.join(scriptFolder,"lines.npy"))

def findBestFeaturesKNN(myFeatures, pca, gp, trainX, trainY):
    print("finding optimal face features KNN")
    # calculate nearest beauty weighted distance to neighbours
    weightedDistances = np.zeros((len(trainX), 1))
    for i in range(len(trainX)):
        neighborFeatures = trainX[i]
        neighborBeauty = trainY[i]
        distanceToNeighbor = np.linalg.norm(myFeatures - neighborFeatures)

        weightedDistance = neighborBeauty / distanceToNeighbor
        weightedDistances[i] = weightedDistance

    nearestWeightsIndexs = np.argsort(weightedDistances, 0)[::-1]

    # find the optimal K size for nearest neighbor
    K = 20
    kNewFeatures = np.zeros((K, len(myFeatures)))
    for k in range(K):
        indexs = nearestWeightsIndexs[:k + 1]
        weights = weightedDistances[indexs]
        features = trainX[indexs]
        kNewFeatures[k, :] = np.sum((weights * features), axis=0) / np.sum(weights)

    y_pred = gp.predict(kNewFeatures)
    bestK = np.argmax(y_pred, 0)

    return kNewFeatures[bestK]

def findBestFeaturesOptimisation(myFeatures, pca, gp):
    print("finding optimal face features optimisation")
    iterCount = 0
    def GPCostFunction(features):
        nonlocal iterCount
        y_pred, sigma2_pred = gp.predict([features], return_std=True)

        iterCount += 1
        if iterCount % 100 == 0:
            print("%i - %0.2f - %0.2f"%(iterCount, y_pred, sigma2_pred))

        return -y_pred / sigma2_pred

    bounds = np.zeros((myFeatures.shape[0],2))
    bounds[:, 0] = myFeatures - 0.15
    bounds[:,1] = myFeatures + 0.15

    optimalNewFaceFeatures = scipy.optimize.minimize(GPCostFunction, myFeatures, method='SLSQP', options={"maxiter":5,"eps":0.001})
    return optimalNewFaceFeatures.x

def findBestFeaturesOptimisation2(myFeatures, pca, gp):
    print("finding optimal face features optimisation")
    iterCount = 0
    def GPCostFunction(features):
        nonlocal iterCount
        y_pred, cov = gp.predict([features], return_cov=True)

        iterCount += 1
        if iterCount % 100 == 0:
            print("%i - %0.2f - %0.2f"%(iterCount, y_pred, cov))

        return -y_pred / cov

    bounds = np.zeros((myFeatures.shape[0],2))
    bounds[:, 0] = myFeatures - 0.15
    bounds[:,1] = myFeatures + 0.15

    optimalNewFaceFeatures = scipy.optimize.minimize(GPCostFunction, myFeatures, method='SLSQP', options={"maxiter":5,"eps":0.001})
    return optimalNewFaceFeatures.x

def findBestFeaturesOptimisation3(myFeatures, pca, gp):
    print("finding optimal face features optimisation")
    iterCount = 0
    def GPCostFunction(features):
        nonlocal iterCount
        y_pred = gp.predict([features])

        reducedFeatures = pca.transform([features])
        LP = np.sum((-np.square(reducedFeatures)) / (2 * pca.explained_variance_))

        iterCount += 1
        if iterCount % 100 == 0:
            print("%i - %0.2f - %0.2f"%(iterCount, y_pred, LP))

        alpha = 0.4
        return (alpha-1)*y_pred - alpha*LP

    bounds = np.zeros((myFeatures.shape[0],2))
    bounds[:, 0] = myFeatures - 0.15
    bounds[:,1] = myFeatures + 0.15

    optimalNewFaceFeatures = scipy.optimize.minimize(GPCostFunction, myFeatures, method='SLSQP', options={"maxiter":5,"eps":0.001})
    return optimalNewFaceFeatures.x

def solveForEyes(oldLandmarks, newLandmarks):
    for eye_points in [LEFT_EYE_POINTS, RIGHT_EYE_POINTS]:
        oldEyeLandmarks = oldLandmarks[eye_points]
        newEyeLandmarks = newLandmarks[eye_points]

        warpMat = cv2.estimateRigidTransform(np.float32(oldEyeLandmarks), np.float32(newEyeLandmarks), fullAffine=False)
        if warpMat is not None:
            warpMat = np.vstack([warpMat, [0, 0, 1]])

            oldEyeLandmarks1 = np.hstack([oldEyeLandmarks, np.ones((oldEyeLandmarks.shape[0],1))])
            transformed = np.zeros(oldEyeLandmarks1.shape)
            for i, landmark in enumerate(oldEyeLandmarks1):
                transformed[i,:] = (np.matrix(warpMat)*landmark[:,np.newaxis]).T

            newLandmarks[eye_points] = transformed[:,:2]
    return newLandmarks

# construct the weighting so that distances between the same features cost more to change
#effectively changing the position of the features in the face, but less of the shape of the feature itself
alphaWeighting = []
for line in faceLines:
    weight = 1
    if line[0] in LEFT_EYE_POINTS and line[1] in LEFT_EYE_POINTS:
        weight = 10
    elif line[0] in RIGHT_EYE_POINTS and line[1] in RIGHT_EYE_POINTS:
        weight = 10
    elif line[0] in MOUTH_POINTS and line[1] in MOUTH_POINTS:
        weight = 10
    elif line[0] in NOSE_POINTS and line[1] in NOSE_POINTS:
        weight = 10
    alphaWeighting.append(weight)
alphaWeighting = np.array(alphaWeighting)
def calculateLandmarksfromFeatures(originalLandmarks, optimalFaceFeatures):
    print("minimising cost of distance constraints from facial features")

    # cost function used to minimize the stress between face features
    def costFunction(landmarks):
        landmarks = np.reshape(landmarks, (-1, 2))
        faceFeatures = landmarks[faceLines[:, 0]] - landmarks[faceLines[:, 1]]
        faceFeatures = np.linalg.norm(faceFeatures, axis=1)

        return np.sum(alphaWeighting * np.square(np.square(faceFeatures) - np.square(optimalFaceFeatures)))

    # find facial landmarks that fit these new distances
    normalizingTerm = getNormalizingFactor(originalLandmarks)
    normLandmarks = originalLandmarks / normalizingTerm

    newLandmarks = scipy.optimize.minimize(costFunction, normLandmarks)
    newLandmarks = np.reshape(newLandmarks.x, (-1, 2)) * normalizingTerm

    newLandmarks = solveForEyes(originalLandmarks, newLandmarks)

    return newLandmarks

def beautifyFace(im, landmarks, features, pca, gp, trainX, trainY, method='KNN'):
    if method=='KNN':
        newFaceFeatures = findBestFeaturesKNN(features, pca, gp, trainX, trainY)
    elif method == 'GP':
        newFaceFeatures = findBestFeaturesOptimisation(features, pca, gp)
    elif method == 'GP2':
        newFaceFeatures = findBestFeaturesOptimisation2(features, pca, gp)
    elif method == 'GP3':
        newFaceFeatures = findBestFeaturesOptimisation3(features, pca, gp)
    # construct the landmarks that satisify the distance constraints of the features
    newLandmarks = calculateLandmarksfromFeatures(landmarks, newFaceFeatures)

    warpedFace = warpFace(im, landmarks, newLandmarks)
    return warpedFace

def compareMethods(im, outpath, us10kpca, ratemepca, us10kgp, ratemegp, us10kTrainX, us10kTrainY, ratemeTrainX, ratemeTrainY):
    landmarks, faceFeatures = getFaceFeatures(im)

    US10KKNN = beautifyFace(im, landmarks, faceFeatures, us10kpca, us10kgp, us10kTrainX, us10kTrainY, method='KNN')
    US10KGP = beautifyFace(im, landmarks, faceFeatures, us10kpca, us10kgp, us10kTrainX, us10kTrainY, method='GP3')
    RateMeKNN = beautifyFace(im, landmarks, faceFeatures, ratemepca, ratemegp, ratemeTrainX, ratemeTrainY, method='KNN')
    RateMeGP = beautifyFace(im, landmarks, faceFeatures, ratemepca, ratemegp, ratemeTrainX, ratemeTrainY, method='GP3')

    displayIm = np.zeros((im.shape[0] * 2, im.shape[1] * 4, im.shape[2]), dtype=np.uint8)
    displayIm[:im.shape[0], :im.shape[1], :] = im.copy()
    displayIm[:im.shape[0], im.shape[1]:im.shape[1] * 2, :] = US10KKNN
    displayIm[:im.shape[0], im.shape[1] * 2:im.shape[1] * 3, :] = US10KGP
    displayIm[im.shape[0]:, im.shape[1]:im.shape[1] * 2, :] = RateMeKNN
    displayIm[im.shape[0]:, im.shape[1] * 2:im.shape[1] * 3, :] = RateMeGP

    diff = np.abs(np.float32(im) - np.float32(US10KGP))
    diff = (diff / np.max(diff)) * 255
    displayIm[:im.shape[0], im.shape[1] * 3:im.shape[1] * 4, :] = np.uint8(diff)

    diff = np.abs(np.float32(im) - np.float32(RateMeGP))
    diff = (diff / np.max(diff)) * 255
    displayIm[im.shape[0]:, im.shape[1] * 3:im.shape[1] * 4, :] = np.uint8(diff)

    cv2.imshow("face", displayIm)
    cv2.waitKey(-1)
    cv2.imwrite(outpath, displayIm)

def beautifyImFromPath(impath, pca, gp, trainX, trainY, method='KNN'):
    im = cv2.imread(impath)
    beautifyIm(im, pca, gp, trainX, trainY, method)

def beautifyIm(im, pca, gp, trainX, trainY, method='KNN'):
    landmarks, faceFeatures = getFaceFeatures(im)
    beautifiedFace = beautifyFace(im, landmarks, faceFeatures, pca, gp, trainX, trainY, method)
    return beautifiedFace

if __name__ == "__main__":
    GENDER = "F"

    dstFolder = "./results/"
    us10kdf = loadUS10KFacialFeatures()
    ratemedf = loadRateMeFacialFeatures()

    us10kgendered = us10kdf.loc[us10kdf['gender'] == GENDER]
    ratemegendered = ratemedf.loc[ratemedf['gender'] == GENDER]

    #split into training sets
    us10kTrainX = np.array(us10kgendered["facefeatures"].as_matrix().tolist())
    us10kTrainY = np.array(us10kgendered["attractiveness"].as_matrix().tolist())

    ratemeTrainX = np.array(ratemegendered["facefeatures"].as_matrix().tolist())
    ratemeTrainY = np.array(ratemegendered["attractiveness"].as_matrix().tolist())

    #load the GP that learnt attractiveness
    ratemepca, ratemegp = loadRateMePCAGP(type="2d", gender=GENDER)
    us10kpca, us10kgp = loadUS10KPCAGP(type="2d", gender=GENDER)

    print("begin beautification")

    im=cv2.imread("C:\\Users\\ellio\\Desktop\\test.png")
    compareMethods(im, ".\\beautified.jpg", us10kpca, ratemepca, us10kgp, ratemegp, us10kTrainX, us10kTrainY, ratemeTrainX, ratemeTrainY)

    # # for each of the test set, make them more beautiful
    # for t in range(len(testX)):
    #     print("working on face %i"%t)
    #     myLandmarks = testL[t]
    #     myFeatures = testX[t]
    #     myActualScore = testY[t]
    #     myImpath = testI[t]
    #
    #     #morph the face
    #     im = cv2.imread(myImpath)
    #
    #     US10KKNN = beautifyFace(im, myLandmarks, myFeatures, us10kpca, us10kgp, trainX, trainY, method='KNN')
    #     US10KGP = beautifyFace(im, myLandmarks, myFeatures, us10kpca, us10kgp, trainX, trainY, method='GP3')
    #     RateMeKNN = beautifyFace(im, myLandmarks, myFeatures, ratemepca, ratemegp, trainXRateMe, trainYRateMe, method='KNN')
    #     RateMeGP = beautifyFace(im, myLandmarks, myFeatures, ratemepca, ratemegp, trainXRateMe, trainYRateMe, method='GP3')
    #
    #     displayIm = np.zeros((im.shape[0]*2, im.shape[1] * 4, im.shape[2]), dtype=np.uint8)
    #     displayIm[:im.shape[0], :im.shape[1], :] = im.copy()
    #     displayIm[:im.shape[0], im.shape[1]:im.shape[1] * 2, :] = US10KKNN
    #     displayIm[:im.shape[0], im.shape[1] * 2:im.shape[1] * 3, :] = US10KGP
    #     displayIm[im.shape[0]:, im.shape[1]:im.shape[1] * 2, :] = RateMeKNN
    #     displayIm[im.shape[0]:, im.shape[1] * 2:im.shape[1] * 3, :] = RateMeGP
    #
    #     diff = np.abs(np.float32(im) - np.float32(US10KGP))
    #     diff = (diff / np.max(diff)) * 255
    #     displayIm[:im.shape[0], im.shape[1] * 3:im.shape[1] * 4, :] = np.uint8(diff)
    #
    #     diff = np.abs(np.float32(im) - np.float32(RateMeGP))
    #     diff = (diff / np.max(diff)) * 255
    #     displayIm[im.shape[0]:, im.shape[1] * 3:im.shape[1] * 4, :] = np.uint8(diff)
    #
    #     cv2.imshow("face", displayIm)
    #     cv2.waitKey(1)
    #     cv2.imwrite(os.path.join(dstFolder, "%04d.jpg" % t), displayIm)
    #
    #     cv2.imwrite(os.path.join(os.path.join(dstFolder, "gp_compare"), "%04d_1.jpg" % t), im)
    #     cv2.imwrite(os.path.join(os.path.join(dstFolder, "gp_compare"), "%04d_2.jpg" % t), US10KGP)