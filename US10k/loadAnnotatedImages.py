import csv
import os
import cv2
import dlib
import numpy as np
import scipy
from sklearn import gaussian_process
from warpTriangles import warpTriangle, getCornerTrianglePts
import pickle

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

#quickly change gender settings
GENDER_WOMEN = 0
GENDER_MEN = 1
GENDER_TYPE = GENDER_MEN
if GENDER_TYPE == GENDER_MEN:
    dstFolder = "./men/"
else:
    dstFolder = "./women/"

#US10K data location
demographicscsv = "E:\\Facedata\\10k US Adult Faces Database\\Full Attribute Scores\\demographic & others labels\\demographic-others-labels-final.csv"
imfolder = "E:\\Facedata\\10k US Adult Faces Database\\Face Images"
#initialize dlib detector
FACESWAP_SHAPEPREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(FACESWAP_SHAPEPREDICTOR_PATH)
#load in the Delaunay triangulation of the face datastructures
faceLines = np.load("lines.npy")
faceTriangles = np.load("triangles.npy")
cornerTriangles = np.load("cornerTriangles.npy")


def readUS10kDemographics():
    print("reading US10K demographics data")
    demographicsData = []
    with open(demographicscsv, 'r') as demoF:
        reader = csv.reader(demoF)
        header = []
        for i, row in enumerate(reader):
            if i == 0:
                header = row
                continue

            row[0] = os.path.join(imfolder, row[0])
            rowDict = {header[j]:row[j] for j in range(len(row)) if os.path.isfile(row[0])}
            demographicsData.append(rowDict)
    return demographicsData

def getFacialFeatures(demographicsData):
    allFaceLandmarks = []
    allFaceFeatures = []
    allAttractiveness = []
    demographicsIndex = []

    print("calculating face features")
    if os.path.isfile(os.path.join(dstFolder,"FaceFeatures.npy")):
        allFaceLandmarks = np.load(os.path.join(dstFolder,"FaceLandmarks.npy"))
        allFaceFeatures = np.load(os.path.join(dstFolder,"FaceFeatures.npy"))
        allAttractiveness = np.load(os.path.join(dstFolder,"FaceAttractiveness.npy"))
        demographicsIndex = np.load(os.path.join(dstFolder,"DemographicsIndex.npy"))
    else:
        for i, data in enumerate(demographicsData):
            attractiveScore = float(data['Attractive'])
            gender = int(data['Gender'])
            if gender == GENDER_TYPE and not np.isnan(attractiveScore):
                im = cv2.imread(data['Filename'])

                rects = detector(im, 1)
                if len(rects) == 0:
                    continue

                landmarks = np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
                landmarks = np.array([[p[0, 0], p[0, 1]] for p in landmarks])

                # normalizingTerm = (landmarks.max(axis=0) - landmarks.min(axis=0))
                normalizingTerm = np.linalg.norm(landmarks[0]-landmarks[16])    #facewidth
                normLandmarks = landmarks / normalizingTerm

                faceFeatures = normLandmarks[faceLines[:,0]] - normLandmarks[faceLines[:,1]]
                faceFeatures = np.linalg.norm(faceFeatures, axis=1)

                #paper normalizes distances using square root of face area

                allFaceLandmarks.append(landmarks)
                allFaceFeatures.append(faceFeatures)
                allAttractiveness.append(attractiveScore)
                demographicsIndex.append(i)

        allFaceLandmarks = np.array(allFaceLandmarks)
        allFaceFeatures = np.array(allFaceFeatures)
        allAttractiveness = np.array(allAttractiveness)
        demographicsIndex = np.array(demographicsIndex)
        np.save(os.path.join(dstFolder,"FaceLandmarks.npy"), allFaceLandmarks)
        np.save(os.path.join(dstFolder,"FaceFeatures.npy"), allFaceFeatures)
        np.save(os.path.join(dstFolder,"FaceAttractiveness.npy"), allAttractiveness)
        np.save(os.path.join(dstFolder,"DemographicsIndex.npy"), demographicsIndex)

    return allFaceLandmarks, allFaceFeatures, allAttractiveness, demographicsIndex

def trainGP(trainX, trainY):
    print("training GP")
    if os.path.isfile(os.path.join(dstFolder,"GP.p")):
        gp = pickle.load(open("GP.p", "rb"))
    else:
        gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
        gp.fit(trainX, trainY)
        pickle.dump(gp, open(os.path.join(dstFolder,"GP.p"), "wb"))
    return gp

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

    y_pred, sigma2_pred = gp.predict(kNewFeatures, eval_MSE=True)
    bestK = np.argmax(y_pred, 0)

    return kNewFeatures[bestK]

def findBestFeaturesOptimisation(myFeatures, gp):
    print("finding optimal face features optimisation")
    iterCount = 0
    def GPCostFunction(faceFeatures):
        nonlocal iterCount
        y_pred, sigma2_pred = gp.predict([faceFeatures], eval_MSE=True)

        iterCount += 1
        if iterCount % 100 == 0:
            print("%i - %0.2f - %0.2f"%(iterCount, myActualScore, y_pred))

        return (1-(y_pred/5)) + sigma2_pred

    bounds = np.zeros((len(myFeatures),2))
    bounds[:, 0] = myFeatures - 0.1
    bounds[:,1] = myFeatures + 0.1

    optimalNewFaceFeatures = scipy.optimize.minimize(GPCostFunction, myFeatures, method='SLSQP', bounds=bounds, options={"maxiter":5,"eps":0.001})
    return optimalNewFaceFeatures.x

def findBestFeaturesBiggerNose(myFeatures):
    print("finding optimal face features for a bigger nose")
    noseScaling = []
    for line in faceLines:
        weight = 1
        if line[0] in NOSE_POINTS and line[1] in NOSE_POINTS:
            weight = 1.4
        if line[0] in LEFT_EYE_POINTS and line[1] in LEFT_EYE_POINTS:
            weight = 1.5
        # elif line[0] in RIGHT_EYE_POINTS and line[1] in RIGHT_EYE_POINTS:
        #     weight = 1.2
        # elif line[0] in MOUTH_POINTS and line[1] in MOUTH_POINTS:
        #     weight = 0.9
        noseScaling.append(weight)
    noseScaling = np.array(noseScaling)
    return myFeatures * noseScaling

def calculateLandmarksfromFeatures(originalLandmarks, optimalFaceFeatures):
    print("minimising cost of distance constraints from facial features")

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
        alphaWeighting.append(weight)
    alphaWeighting = np.array(alphaWeighting)

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

    return newLandmarks

def faceWarp(im, oldLandmarks, newLandmarks):
    print("morphing face")
    newIm = im.copy()
    # newIm = np.zeros(im.shape)
    oldCornerTrianglePts = getCornerTrianglePts(im, cornerTriangles, oldLandmarks)
    newCornerTrianglePts = getCornerTrianglePts(im, cornerTriangles, newLandmarks)

    for ti in range(len(newCornerTrianglePts)):
        oldTriangle = oldCornerTrianglePts[ti]
        newTriangle = newCornerTrianglePts[ti]

        newIm = warpTriangle(im, newIm, oldTriangle, newTriangle)

    for triangle in faceTriangles:
        oldTriangle = oldLandmarks[triangle]
        newTriangle = newLandmarks[triangle]

        newIm = warpTriangle(im, newIm, oldTriangle, newTriangle)
    newIm = np.uint8(newIm)
    return newIm

if __name__ == "__main__":
    demographicsData = readUS10kDemographics()
    allFaceLandmarks, allFaceFeatures, allAttractiveness, demographicsIndex = getFacialFeatures(demographicsData)

    #split into training sets
    trainSize = int(len(allFaceFeatures) * 0.8)
    trainX = allFaceFeatures[:trainSize]
    trainY = allAttractiveness[:trainSize]

    testL = allFaceLandmarks[trainSize:]
    testX = allFaceFeatures[trainSize:]
    testY = allAttractiveness[trainSize:]
    testI = demographicsIndex[trainSize:]

    #create a GP that learns attractiveness
    gp = trainGP(trainX, trainY)

    print("begin beautification")
    # for each of the test set, make them more beautiful
    for t in range(len(testX)):
        print("working on face %i"%t)
        myLandmarks = testL[t]
        myFeatures = testX[t]
        myActualScore = testY[t]
        myDemographicsIndex = testI[t]

        #get a set of face features that are more beautiful
        optimalNewFaceFeaturesKNN = findBestFeaturesKNN(myFeatures, gp, trainX, trainY)
        optimalNewFaceFeaturesGP = findBestFeaturesOptimisation(myFeatures, gp)
        # optimalNewFaceFeaturesGP = findBestFeaturesBiggerNose(myFeatures)

        #construct the landmarks that satisify the distance constraints of the features
        newLandmarksKNN = calculateLandmarksfromFeatures(myLandmarks, optimalNewFaceFeaturesKNN)
        newLandmarksGP = calculateLandmarksfromFeatures(myLandmarks, optimalNewFaceFeaturesGP)

        #morph the face
        im = cv2.imread(demographicsData[myDemographicsIndex]['Filename'])

        beautifulFaceKNN = faceWarp(im, myLandmarks, newLandmarksKNN)
        beautifulFaceGP = faceWarp(im, myLandmarks, newLandmarksGP)

        displayIm = np.zeros((im.shape[0], im.shape[1] * 4, im.shape[2]), dtype=np.uint8)
        displayIm[:, :im.shape[1], :] = im.copy()
        displayIm[:, im.shape[1]:im.shape[1] * 2, :] = beautifulFaceKNN.copy()
        displayIm[:, im.shape[1]*2:im.shape[1] * 3, :] = beautifulFaceGP.copy()

        # draw the landmarks
        for i, landmark in enumerate(newLandmarksKNN):
            op = (int(myLandmarks[i][0]), int(myLandmarks[i][1]))
            cv2.circle(im, op, 3, (255, 0, 255), thickness=-1)
            p = (int(landmark[0]), int(landmark[1]))
            cv2.circle(im, p, 3, (0, 255, 255), thickness=-1)
        displayIm[:, im.shape[1] * 3:, :] = im.copy()

        # cv2.imshow("face", displayIm)
        # cv2.waitKey(-1)
        cv2.imwrite(os.path.join(dstFolder,"imgs/%04d.jpg" % t), displayIm)