import numpy as np
import cv2
from Beautifier.beautifier import findBestFeaturesKNN, LEFT_EYE_POINTS, RIGHT_EYE_POINTS, MOUTH_POINTS, NOSE_POINTS
from Beautifier.warpFace import warpFace, drawLandmarks
from Beautifier.faceFeatures import getLandmarks
from Beautifier.face3D.faceFeatures3D import getFaceFeatures3D, getMeshFromLandmarks, model, blendshapes, getFaceFeatures3D2DFromMesh, newFaceLines, vertIndices
from Beautifier.face3D.warpFace3D import warpFace3D, projectVertsTo2D
import eos
import os
import scipy

scriptFolder = os.path.dirname(os.path.realpath(__file__))

alphaWeighting = []
for line in newFaceLines:
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
        landmarks = np.reshape(landmarks, (-1, 3))
        faceFeatures = landmarks[newFaceLines[:, 0]] - landmarks[newFaceLines[:, 1]]
        faceFeatures = np.linalg.norm(faceFeatures, axis=1)

        return np.sum(alphaWeighting * np.square(np.square(faceFeatures) - np.square(optimalFaceFeatures)))

    # find facial landmarks that fit these new distances
    newLandmarks = scipy.optimize.minimize(costFunction, originalLandmarks[:,:3])
    newLandmarks = np.reshape(newLandmarks.x, (-1, 3))

    newLandmarks = np.hstack([newLandmarks, np.ones((newLandmarks.shape[0],1), dtype=np.float64)])

    return newLandmarks

def findBestFeaturesFudge(features3D):
    features3D[2] -= 2
    features3D[3] -= 1
    return features3D

def findBestFeaturesOptimisation(features3D, gp):
    print("finding optimal face features optimisation")

    def GPCostFunction(features):
        y_pred = gp.predict([features])
        return -y_pred

    bounds = np.zeros((features3D.shape[0],2))
    bounds[:, 0] = -200
    bounds[:, 1] = 200

    optimalNewFaceFeatures = scipy.optimize.minimize(GPCostFunction, features3D, bounds=bounds, method='SLSQP', options={"maxiter": 5, "eps": 0.001})
    return optimalNewFaceFeatures.x

def beautifyFace3D(im, landmarks, pca, gp, trainX, trainY, method='KNN', exaggeration=0.5):
    mesh, pose, shape_coeffs, blendshape_coeffs = getMeshFromLandmarks(landmarks, im, num_shape_coefficients_to_fit=-1)
    features3D = np.array(shape_coeffs)

    if method=='KNN':
        newFaceFeatures = findBestFeaturesKNN(features3D, pca, gp, trainX, trainY)
    elif method == 'GP':
        newFaceFeatures = findBestFeaturesOptimisation(features3D, gp)
    elif method == 'fudge it':
        newFaceFeatures = findBestFeaturesFudge(features3D)

    delta = newFaceFeatures - features3D
    newFaceFeatures = features3D + (delta*exaggeration)

    newMesh = eos.morphablemodel.draw_sample(model, blendshapes, newFaceFeatures, blendshape_coeffs,[])#newFaceFeatures[63:], [])

    warpedIm = warpFace3D(im, mesh, pose, newMesh)
    return warpedIm

def beautifyFace3D2D(im, landmarks, pca, gp, trainX, trainY, method='KNN', exaggeration=0.5):
    mesh, pose, shape_coeffs, blendshape_coeffs = getMeshFromLandmarks(landmarks, im, num_shape_coefficients_to_fit=10)
    landmarks3D, features3D = getFaceFeatures3D2DFromMesh(mesh)

    if method=='KNN':
        newFaceFeatures = findBestFeaturesKNN(features3D, pca, gp, trainX, trainY)
    elif method == 'GP':
        newFaceFeatures = findBestFeaturesOptimisation(features3D, gp)
    elif method == 'fudge it':
        newFaceFeatures = findBestFeaturesFudge(features3D)

    # construct the landmarks that satisify the distance constraints of the features
    newLandmarks = calculateLandmarksfromFeatures(landmarks3D, newFaceFeatures)

    newLandmarks2D = projectVertsTo2D(newLandmarks, pose, im)
    newLandmarks2D = np.array([landmarks[i] if vertIndices[i] == -1 else newLandmarks2D[i] for i in range(len(vertIndices))]).astype(np.int32)

    newMesh, pose, shape_coeffs, blendshape_coeffs = getMeshFromLandmarks(newLandmarks2D, im, num_shape_coefficients_to_fit=10)

    old = drawLandmarks(im, landmarks)
    new = drawLandmarks(im, newLandmarks2D, color=(0,255,255))
    stuff = drawLandmarks(old, newLandmarks2D, color=(0,255,255))
    cv2.imshow("old", old)
    cv2.imshow("new", new)
    cv2.imshow("oldC", stuff)
    cv2.waitKey(-1)

    warpedIm = warpFace(im, landmarks, newLandmarks2D)
    return warpedIm

def compareMethods(im, datasets):
    landmarks = getLandmarks(im)

    displayIm = np.zeros((im.shape[0] * len(datasets), im.shape[1] * 4, im.shape[2]), dtype=np.uint8)
    displayIm[:im.shape[0], :im.shape[1], :] = im.copy()
    for i, dataset in enumerate(datasets):
        trainX, trainY, pca, gp = dataset

        KNN = beautifyFace3D2D(im, landmarks, pca, gp, trainX, trainY, method='KNN')
        GP = beautifyFace3D2D(im, landmarks, pca, gp, trainX, trainY, method='GP')

        displayIm[im.shape[0] * i:im.shape[0] * (i + 1), im.shape[1]:im.shape[1] * 2, :] = KNN
        displayIm[im.shape[0] * i:im.shape[0] * (i + 1), im.shape[1] * 2:im.shape[1] * 3, :] = GP

        diff = np.abs(np.float32(im) - np.float32(GP))
        diff = (diff / np.max(diff)) * 255
        displayIm[im.shape[0] * i:im.shape[0] * (i + 1), im.shape[1] * 3:im.shape[1] * 4, :] = np.uint8(diff)

    return displayIm

def beautifyIm3DFromPath(impath, pca, gp, trainX, trainY, method='KNN'):
    im = cv2.imread(impath)
    beautifyIm3D(im, pca, gp, trainX, trainY, method)

def beautifyIm3D(im, pca, gp, trainX, trainY, method='KNN'):
    landmarks = getLandmarks(im)
    beautifiedFace = beautifyFace3D(im, landmarks, pca, gp, trainX, trainY, method)
    return beautifiedFace

def rateFace3D(im, pca, gp):
    landmarks = getLandmarks(im)
    faceFeatures = getFaceFeatures3D([im],[landmarks], num_shape_coefficients_to_fit=10)

    return gp.predict(faceFeatures)[0]

if __name__ == "__main__":
    import glob
    from US10k.US10k import loadUS10k
    from RateMe.RateMe import loadRateMe

    GENDER = "F"

    dstFolder = "./results3F/"

    datasets = [#loadUS10k(type="3d", gender=GENDER),
                loadRateMe(type="3d2d", gender=GENDER)]

    print("begin beautification")
    import glob

    for i, impath in enumerate(glob.glob(r"C:\Users\Elliot\Desktop\*.png")):
        im = cv2.imread(impath)
        filename = os.path.basename(impath)

        comparedIm = compareMethods(im, datasets)

        cv2.imshow("face", comparedIm)
        cv2.waitKey(1)
        cv2.imwrite(os.path.join(dstFolder, filename), comparedIm)

        compareIndex = 1
        cv2.imwrite(os.path.join(dstFolder, "compare/%i_1.jpg" % i), im)
        cv2.imwrite(os.path.join(dstFolder, "compare/%i_2.jpg" % i), comparedIm[im.shape[0] * compareIndex:im.shape[0] * (compareIndex + 1), im.shape[1] * 2:im.shape[1] * 3, :])