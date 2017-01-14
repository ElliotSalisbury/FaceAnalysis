import numpy as np
import cv2
from Beautifier.beautifier import findBestFeaturesKNN
from Beautifier.faceFeatures import getLandmarks
from Beautifier.face3D.faceFeatures3D import getFaceFeatures3D, getMeshFromLandmarks, model, blendshapes
from Beautifier.face3D.warpFace3D import warpFace3D
import eos
import os
import scipy

scriptFolder = os.path.dirname(os.path.realpath(__file__))

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
    bounds[:, 0] = -2
    bounds[:, 1] = 2

    optimalNewFaceFeatures = scipy.optimize.minimize(GPCostFunction, features3D, bounds=bounds, method='SLSQP', options={"maxiter": 5, "eps": 0.001})
    return optimalNewFaceFeatures.x

def beautifyFace3D(im, landmarks, pca, gp, trainX, trainY, method='KNN'):
    mesh, pose, shape_coeffs, blendshape_coeffs = getMeshFromLandmarks(landmarks, im, num_shape_coefficients_to_fit=10)
    features3D = np.array(shape_coeffs)

    if method=='KNN':
        newFaceFeatures = findBestFeaturesKNN(features3D, pca, gp, trainX, trainY)
    elif method == 'GP':
        newFaceFeatures = findBestFeaturesOptimisation(features3D, gp)
    elif method == 'fudge it':
        newFaceFeatures = findBestFeaturesFudge(features3D)

    newMesh = eos.morphablemodel.draw_sample(model, blendshapes, newFaceFeatures, blendshape_coeffs,[])#newFaceFeatures[63:], [])

    warpedIm = warpFace3D(im, mesh, pose, newMesh)
    return warpedIm

def compareMethods(im, datasets):
    landmarks = getLandmarks(im)

    displayIm = np.zeros((im.shape[0] * len(datasets), im.shape[1] * 4, im.shape[2]), dtype=np.uint8)
    displayIm[:im.shape[0], :im.shape[1], :] = im.copy()
    for i, dataset in enumerate(datasets):
        trainX, trainY, pca, gp = dataset

        KNN = beautifyFace3D(im, landmarks, pca, gp, trainX, trainY, method='KNN')
        GP = beautifyFace3D(im, landmarks, pca, gp, trainX, trainY, method='GP')

        displayIm[im.shape[0] * i:im.shape[0] * (i + 1), im.shape[1]:im.shape[1] * 2, :] = KNN
        displayIm[im.shape[0] * i:im.shape[0] * (i + 1), im.shape[1] * 2:im.shape[1] * 3, :] = GP

        diff = np.abs(np.float32(im) - np.float32(GP))
        diff = (diff / np.max(diff)) * 255
        displayIm[:im.shape[0], im.shape[1] * 3:im.shape[1] * 4, :] = np.uint8(diff)

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

    GENDER = "M"

    dstFolder = "./results3M/"
    us10kdf = loadUS10k(type="3d", gender=GENDER)

    datasets = [us10kdf]

    print("begin beautification")
    import glob

    for i, impath in enumerate(glob.glob("E:\\Facedata\\10k US Adult Faces Database\\Publication Friendly 49-Face Database\\49 Face Images\\*.jpg")):
        im = cv2.imread(impath)
        filename = os.path.basename(impath)
        comparedIm = compareMethods(im, datasets)

        cv2.imshow("face", comparedIm)
        cv2.waitKey(1)
        cv2.imwrite(os.path.join(dstFolder, filename), comparedIm)

        cv2.imwrite(os.path.join(dstFolder, "compare/%i_1.jpg" % i), im)
        cv2.imwrite(os.path.join(dstFolder, "compare/%i_2.jpg" % i), comparedIm[im.shape[0] * 0:im.shape[0] * (0 + 1), im.shape[1] * 2:im.shape[1] * 3, :])