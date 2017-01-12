import numpy as np
import cv2
from US10k.US10k import loadUS10k
from RateMe.RateMe import loadRateMe
from Beautifier.beautifier import findBestFeaturesKNN, findBestFeaturesOptimisation, findBestFeaturesOptimisation2, findBestFeaturesOptimisation3
from Beautifier.faceFeatures import getLandmarks
from Beautifier.face3D.faceFeatures3D import getFaceFeatures3D, getMeshFromLandmarks, model, blendshapes
from Beautifier.face3D.warpFace3D import warpFace3D
import eos
import os

scriptFolder = os.path.dirname(os.path.realpath(__file__))

def findBestFeaturesFudge(features3D):
    features3D[2] -= 2
    features3D[3] -= 1
    return features3D

def beautifyFace3D(im, landmarks, pca, gp, trainX, trainY, method='KNN', exaggeration=1.5):
    mesh, pose, shape_coeffs, blendshape_coeffs = getMeshFromLandmarks(landmarks, im, num_shape_coefficients_to_fit=10)
    features3D = np.array(shape_coeffs)

    if method=='KNN':
        newFaceFeatures = findBestFeaturesKNN(features3D, pca, gp, trainX, trainY)
    elif method == 'GP':
        newFaceFeatures = findBestFeaturesOptimisation(features3D, pca, gp)
    elif method == 'GP2':
        newFaceFeatures = findBestFeaturesOptimisation2(features3D, pca, gp)
    elif method == 'GP3':
        newFaceFeatures = findBestFeaturesOptimisation3(features3D, pca, gp)
    elif method == 'fudge it':
        newFaceFeatures = findBestFeaturesFudge(features3D)

    delta = newFaceFeatures - features3D
    newFaceFeatures = features3D + exaggeration*delta
    # newFaceFeatures = shape_coeffs
    # newFaceFeatures[0] -= 1.0
    newMesh = eos.morphablemodel.draw_sample(model, blendshapes, newFaceFeatures, blendshape_coeffs,[])#newFaceFeatures[63:], [])

    warpedIm = warpFace3D(im, mesh, pose, newMesh)
    return warpedIm

def compareMethods(im, outpath, us10kpca, ratemepca, us10kgp, ratemegp, us10kTrainX, us10kTrainY, ratemeTrainX, ratemeTrainY):
    landmarks = getLandmarks(im)

    # US10kKNN = beautifyFace3D(im, landmarks, us10kpca, us10kgp, us10kTrainX, us10kTrainY, method='KNN')
    # US10kGP = beautifyFace3D(im, landmarks, us10kpca, us10kgp, us10kTrainX, us10kTrainY, method='GP3')
    RateMeKNN = beautifyFace3D(im, landmarks, ratemepca, ratemegp, ratemeTrainX, ratemeTrainY, method='KNN')
    RateMeGP = beautifyFace3D(im, landmarks, ratemepca, ratemegp, ratemeTrainX, ratemeTrainY, method='GP2')

    displayIm = np.zeros((im.shape[0] * 2, im.shape[1] * 4, im.shape[2]), dtype=np.uint8)
    displayIm[:im.shape[0], :im.shape[1], :] = im.copy()
    # displayIm[:im.shape[0], im.shape[1]:im.shape[1] * 2, :] = US10kKNN
    # displayIm[:im.shape[0], im.shape[1] * 2:im.shape[1] * 3, :] = US10kGP
    displayIm[im.shape[0]:, im.shape[1]:im.shape[1] * 2, :] = RateMeKNN
    displayIm[im.shape[0]:, im.shape[1] * 2:im.shape[1] * 3, :] = RateMeGP

    # diff = np.abs(np.float32(im) - np.float32(US10kGP))
    # diff = (diff / np.max(diff)) * 255
    # displayIm[:im.shape[0], im.shape[1] * 3:im.shape[1] * 4, :] = np.uint8(diff)

    diff = np.abs(np.float32(im) - np.float32(RateMeGP))
    diff = (diff / np.max(diff)) * 255
    displayIm[im.shape[0]:, im.shape[1] * 3:im.shape[1] * 4, :] = np.uint8(diff)

    cv2.imshow("face", displayIm)
    cv2.waitKey(1)
    cv2.imwrite(outpath, displayIm)

def beautifyIm3DFromPath(impath, pca, gp, trainX, trainY, method='KNN'):
    im = cv2.imread(impath)
    beautifyIm3D(im, pca, gp, trainX, trainY, method)

def beautifyIm3D(im, pca, gp, trainX, trainY, method='KNN', exaggeration=1.5):
    landmarks = getLandmarks(im)
    beautifiedFace = beautifyFace3D(im, landmarks, pca, gp, trainX, trainY, method, exaggeration=exaggeration)
    return beautifiedFace

def rateFace3D(im, pca, gp):
    landmarks = getLandmarks(im)
    faceFeatures = getFaceFeatures3D([im],[landmarks], num_shape_coefficients_to_fit=10)

    return gp.predict(faceFeatures)[0]

if __name__ == "__main__":
    import glob
    from Beautifier.face3D.faceFeatures3D import ensureImageLessThanMax
    from Beautifier.beautifier import beautifyIm

    GENDER = "F"

    US10k_3D = loadRateMe(type="3d", gender=GENDER)
    trainX, trainY, pca, gp = US10k_3D

    im = cv2.imread("C:\\Users\\ellio\\Desktop\\lena.bmp")
    im = ensureImageLessThanMax(im, maxsize=1024)

    rating = rateFace3D(im, pca, gp)
    print("Rating: %f"%rating)
    better = beautifyIm3D(im, pca, gp, trainX, trainY, method='KNN', exaggeration=20)

    cv2.imshow("b", better)
    cv2.imshow("o", im)
    cv2.waitKey(-1)

    # print("begin beautification")
    #
    # for impath in glob.glob("E:\\Facedata\\10k US Adult Faces Database\\Publication Friendly 49-Face Database\\49 Face Images\\*.jpg"):
    #     im = cv2.imread(impath)
    #     compareMethods(im, impath+".beautified.jpg", us10kpca, ratemepca, us10kgp, ratemegp, us10kTrainX, us10kTrainY,ratemeTrainX, ratemeTrainY)