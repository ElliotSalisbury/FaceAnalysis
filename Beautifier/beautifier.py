import numpy as np
import cv2
import pickle
import os
import scipy
from sklearn import gaussian_process
from warpFace import warpFace
from US10K import loadUS10KFacialFeatures
from calculateFaceData import loadRateMeFacialFeatures

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

faceLines = np.load("C:\\Users\\ellio\\PycharmProjects\\circlelines\\Beautifier\\lines.npy")

def findBestFeaturesKNN(myFeatures, gp, trainX, trainY):
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

def findBestFeaturesOptimisation(myFeatures, gp):
    print("finding optimal face features optimisation")
    iterCount = 0
    def GPCostFunction(faceFeatures):
        nonlocal iterCount
        y_pred, sigma2_pred = gp.predict([faceFeatures], return_std=True)

        iterCount += 1
        if iterCount % 100 == 0:
            print("%i - %0.2f - %0.2f"%(iterCount, y_pred, sigma2_pred))

        return (1-(y_pred/5)) + sigma2_pred

    bounds = np.zeros((len(myFeatures),2))
    bounds[:, 0] = myFeatures - 0.1
    bounds[:,1] = myFeatures + 0.1

    optimalNewFaceFeatures = scipy.optimize.minimize(GPCostFunction, myFeatures, method='SLSQP', bounds=bounds, options={"maxiter":5,"eps":0.001})
    return optimalNewFaceFeatures.x

def solveForEyes(oldLandmarks, newLandmarks):
    for eye_points in [LEFT_EYE_POINTS, RIGHT_EYE_POINTS]:
        oldEyeLandmarks = oldLandmarks[eye_points]
        newEyeLandmarks = newLandmarks[eye_points]

        warpMat = cv2.estimateRigidTransform(np.float32(oldEyeLandmarks), np.float32(newEyeLandmarks), fullAffine=False)
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
    normalizingTerm = np.linalg.norm(originalLandmarks[0] - originalLandmarks[16])  # facewidth
    normLandmarks = originalLandmarks / normalizingTerm

    newLandmarks = scipy.optimize.minimize(costFunction, normLandmarks)
    newLandmarks = np.reshape(newLandmarks.x, (-1, 2)) * normalizingTerm

    newLandmarks = solveForEyes(originalLandmarks, newLandmarks)

    return newLandmarks

if __name__ == "__main__":
    dstFolder = "./results/"
    us10kdf = loadUS10KFacialFeatures()
    ratemedf = loadRateMeFacialFeatures()

    us10kwomen = us10kdf.loc[us10kdf['gender'] == 'F']
    ratemewomen = ratemedf.loc[ratemedf['gender'] == 'F']
    ratemewomen = ratemewomen.loc[ratemewomen['attractiveness'] >= 8]

    #split into training sets
    trainSize = int(us10kwomen.shape[0] * 0.8)
    traindf = us10kwomen[:trainSize]
    trainX = np.array(traindf["facefeatures"].as_matrix().tolist())
    trainY = np.array(traindf["attractiveness"].as_matrix().tolist())

    testdf = us10kwomen[trainSize:]
    testX = np.array(testdf["facefeatures"].as_matrix().tolist())
    testY = np.array(testdf["attractiveness"].as_matrix().tolist())
    testL = np.array(testdf["landmarks"].as_matrix().tolist())
    testI = np.array(testdf["impath"].as_matrix().tolist())

    trainXRateMe = np.array(ratemewomen["facefeatures"].as_matrix().tolist())
    trainYRateMe = np.array(ratemewomen["attractiveness"].as_matrix().tolist())

    #load the GP that learnt attractiveness
    ratemegp = pickle.load(
        open("C:\\Users\\ellio\\PycharmProjects\\circlelines\\rRateMe\\GP_F.p", "rb"))
    us10kgp = pickle.load(
        open("C:\\Users\\ellio\\PycharmProjects\\circlelines\\US10k\\GP_F.p", "rb"))

    print("begin beautification")
    # for each of the test set, make them more beautiful
    for t in range(len(testX)):
        print("working on face %i"%t)
        myLandmarks = testL[t]
        myFeatures = testX[t]
        myActualScore = testY[t]
        myImpath = testI[t]

        #get a set of face features that are more beautiful
        optimalNewFaceFeaturesKNN = findBestFeaturesKNN(myFeatures, us10kgp, trainX, trainY)
        # optimalNewFaceFeaturesGP = findBestFeaturesOptimisation(myFeatures, us10kgp)

        #construct the landmarks that satisify the distance constraints of the features
        newLandmarksKNN = calculateLandmarksfromFeatures(myLandmarks, optimalNewFaceFeaturesKNN)
        # newLandmarksGP = calculateLandmarksfromFeatures(myLandmarks, optimalNewFaceFeaturesGP)

        #morph the face
        im = cv2.imread(myImpath)

        beautifulFaceKNN = warpFace(im, myLandmarks, newLandmarksKNN)
        # beautifulFaceGP = warpFace(im, myLandmarks, newLandmarksGP)

        displayIm = np.zeros((im.shape[0]*2, im.shape[1] * 4, im.shape[2]), dtype=np.uint8)
        displayIm[:im.shape[0], :im.shape[1], :] = im.copy()
        displayIm[:im.shape[0], im.shape[1]:im.shape[1] * 2, :] = beautifulFaceKNN.copy()
        # displayIm[:im.shape[0], im.shape[1]*2:im.shape[1] * 3, :] = beautifulFaceGP.copy()


        #TEST RATE ME
        # get a set of face features that are more beautiful
        optimalNewFaceFeaturesKNN = findBestFeaturesKNN(myFeatures, ratemegp, trainXRateMe, trainYRateMe)
        # optimalNewFaceFeaturesGP = findBestFeaturesOptimisation(myFeatures, ratemegp)
        # optimalNewFaceFeaturesGP = findBestFeaturesBiggerNose(myFeatures)

        # construct the landmarks that satisify the distance constraints of the features
        newLandmarksKNN = calculateLandmarksfromFeatures(myLandmarks, optimalNewFaceFeaturesKNN)
        # newLandmarksGP = calculateLandmarksfromFeatures(myLandmarks, optimalNewFaceFeaturesGP)

        beautifulFaceKNN = warpFace(im, myLandmarks, newLandmarksKNN)
        # beautifulFaceGP = warpFace(im, myLandmarks, newLandmarksGP)
        displayIm[im.shape[0]:, im.shape[1]:im.shape[1] * 2, :] = beautifulFaceKNN.copy()
        # displayIm[im.shape[0]:, im.shape[1] * 2:im.shape[1] * 3, :] = beautifulFaceGP.copy()

        # cv2.imshow("face", displayIm)
        # cv2.waitKey(-1)
        cv2.imwrite(os.path.join(dstFolder,"%04d.jpg" % t), displayIm)