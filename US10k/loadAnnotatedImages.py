import csv
import os
import cv2
import dlib
import numpy as np
import scipy
from sklearn import gaussian_process
from warpTriangles import warpTriangle

demographicscsv = "E:\\Facedata\\10k US Adult Faces Database\\Full Attribute Scores\\demographic & others labels\\demographic-others-labels-final.csv"
imfolder = "E:\\Facedata\\10k US Adult Faces Database\\Face Images"

print("reading data")
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

FACESWAP_SHAPEPREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(FACESWAP_SHAPEPREDICTOR_PATH)
faceLines = np.load("lines.npy")
faceTriangles = np.load("triangles.npy")

allFaceLandmarks = []
allFaceFeatures = []
allAttractiveness = []
demographicsIndex = []

print("calculating face features")
if os.path.isfile("./FaceFeatures.npy"):
    allFaceLandmarks = np.load("FaceLandmarks.npy")
    allFaceFeatures = np.load("FaceFeatures.npy")
    allAttractiveness = np.load("FaceAttractiveness.npy")
    demographicsIndex = np.load("DemographicsIndex.npy")
else:
    for i, data in enumerate(demographicsData):
        attractiveScore = float(data['Attractive'])
        gender = int(data['Gender'])
        if gender == 0 and not np.isnan(attractiveScore):
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
    np.save("FaceLandmarks.npy", allFaceLandmarks)
    np.save("FaceFeatures.npy", allFaceFeatures)
    np.save("FaceAttractiveness.npy", allAttractiveness)
    np.save("DemographicsIndex.npy", demographicsIndex)

print("training GP")
trainSize = int(len(allFaceFeatures)*0.8)
trainX = allFaceFeatures[:trainSize]
trainY = allAttractiveness[:trainSize]

testL = allFaceLandmarks[trainSize:]
testX = allFaceFeatures[trainSize:]
testY = allAttractiveness[trainSize:]
testI = demographicsIndex[trainSize:]

gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
gp.fit(trainX, trainY)

print("predicting attractiveness")
y_pred, sigma2_pred = gp.predict(testX, eval_MSE=True)
#
# print("copying images")
# sortedIndexs = np.argsort(y_pred,0)
# for i, index in enumerate(sortedIndexs):
#     demoIndex = testI[index]
#     impath = demographicsData[demoIndex]['Filename']
#     actual = float(demographicsData[demoIndex]['Attractive'])
#     predicted = y_pred[index]
#
#     im = cv2.imread(impath)
#     cv2.imwrite("./imgs/%d_%0.2f_%0.2f.jpg"%(i,actual,predicted), im)

print("beautification KNN")
#for all of the training set
for t in range(len(testX)):
    myLandmarks = testL[t]
    myFeatures = testX[t]
    # myPredScore = y_pred[t]
    myActualScore = testY[t]
    myDemographicsIndex = testI[t]

    print("finding optimal face")
    #calculate nearest beauty weighted distance to neighbours
    weightedDistances = np.zeros((len(trainX),1))
    for i in range(len(trainX)):
        neighborFeatures = trainX[i]
        neighborBeauty = trainY[i]
        distanceToNeighbor = np.linalg.norm(myFeatures - neighborFeatures)

        weightedDistance = neighborBeauty / distanceToNeighbor
        weightedDistances[i] = weightedDistance

    nearestWeightsIndexs = np.argsort(weightedDistances, 0)[::-1]

    #find the optimal K size for nearest neighbor
    K = 20
    kNewFeatures = np.zeros((K, len(myFeatures)))
    for k in range(K):
        indexs = nearestWeightsIndexs[:k+1]
        weights = weightedDistances[indexs]
        features = trainX[indexs]
        kNewFeatures[k,:] = np.sum((weights * features), axis=0) / np.sum(weights)

    y_pred, sigma2_pred = gp.predict(kNewFeatures, eval_MSE=True)
    bestK = np.argmax(y_pred,0)

    optimalNewFaceFeatures = kNewFeatures[bestK]

    print("minimising cost of facial features morphing")
    #cost function used to minimize the stress between face features
    alphaWeighting = np.ones((len(faceLines), 1))
    def costFunction(landmarks):
        landmarks = np.reshape(landmarks, (-1,2))
        faceFeatures = landmarks[faceLines[:, 0]] - landmarks[faceLines[:, 1]]
        faceFeatures = np.linalg.norm(faceFeatures, axis=1)

        return np.sum(np.square(np.square(faceFeatures) - np.square(optimalNewFaceFeatures)))

    #find facial landmarks that fit these new distances
    normalizingTerm = np.linalg.norm(myLandmarks[0] - myLandmarks[16])  # facewidth
    normLandmarks = myLandmarks / normalizingTerm

    alphaWeighting * np.square(np.square(myFeatures) -  np.square(optimalNewFaceFeatures))
    newLandmarks = scipy.optimize.minimize(costFunction, normLandmarks)
    newLandmarks = np.reshape(newLandmarks.x, (-1, 2)) * normalizingTerm

    print("morphing face")
    im = cv2.imread(demographicsData[myDemographicsIndex]['Filename'])

    # for i, landmark in enumerate(newLandmarks):
    #     p = (int(landmark[0]), int(landmark[1]))
    #     op = (int(myLandmarks[i][0]), int(myLandmarks[i][1]))
    #     cv2.circle(im, p, 3, (0, 255, 255), thickness=-1)
    #     cv2.circle(im, op, 3, (255, 0, 255), thickness=-1)
    # cv2.imshow("face", im)
    # cv2.waitKey(-1)

    newIm = im.copy()
    for triangle in faceTriangles:
        oldTriangle = myLandmarks[triangle]
        newTriangle = newLandmarks[triangle]

        newIm = warpTriangle(im, newIm, oldTriangle, newTriangle)
    newIm = np.uint8(newIm)

    displayIm = np.zeros((im.shape[0], im.shape[1]*2, im.shape[2]), dtype=np.uint8)
    displayIm[:, :im.shape[1], :] = im.copy()
    displayIm[:, im.shape[1]:, :] = newIm.copy()
    cv2.imshow("faced", displayIm)
    cv2.waitKey(-1)
