import numpy as np
import cv2
from US10k.US10K import loadUS10KFacialFeatures, loadUS10KPCAGP
from RateMe.RateMe import loadRateMeFacialFeatures, loadRateMePCAGP
from Beautifier.beautifier import findBestFeaturesKNN, findBestFeaturesOptimisation, findBestFeaturesOptimisation2, findBestFeaturesOptimisation3
from Beautifier.faceFeatures import getLandmarks
from Beautifier.face3D.faceFeatures3D import getFaceFeatures3D, getMeshFromLandmarks, model, blendshapes
from Beautifier.face3D.warpFace3D import warpFace3D
import eos
import os

scriptFolder = os.path.dirname(os.path.realpath(__file__))

def beautifyFace3D(im, landmarks, pca, gp, trainX, trainY, method='KNN'):
    mesh, pose, shape_coeffs, blendshape_coeffs = getMeshFromLandmarks(landmarks, im)
    features3D = np.array(shape_coeffs + blendshape_coeffs)

    if method=='KNN':
        newFaceFeatures = findBestFeaturesKNN(features3D, pca, gp, trainX, trainY)
    elif method == 'GP':
        newFaceFeatures = findBestFeaturesOptimisation(features3D, pca, gp)
    elif method == 'GP2':
        newFaceFeatures = findBestFeaturesOptimisation2(features3D, pca, gp)
    elif method == 'GP3':
        newFaceFeatures = findBestFeaturesOptimisation3(features3D, pca, gp)

    # newFaceFeatures = shape_coeffs
    # newFaceFeatures[0] -= 1.0
    newMesh = eos.morphablemodel.draw_sample(model, blendshapes, newFaceFeatures[:63], blendshape_coeffs,[])#newFaceFeatures[63:], [])

    warpedIm = warpFace3D(im, mesh, pose, newMesh)
    return warpedIm

def compareMethods(im, outpath, us10kpca, ratemepca, us10kgp, ratemegp, us10kTrainX, us10kTrainY, ratemeTrainX, ratemeTrainY):
    landmarks = getLandmarks(im)

    US10KKNN = beautifyFace3D(im, landmarks, us10kpca, us10kgp, us10kTrainX, us10kTrainY, method='KNN')
    US10KGP = beautifyFace3D(im, landmarks, us10kpca, us10kgp, us10kTrainX, us10kTrainY, method='GP3')
    RateMeKNN = beautifyFace3D(im, landmarks, ratemepca, ratemegp, ratemeTrainX, ratemeTrainY, method='KNN')
    RateMeGP = beautifyFace3D(im, landmarks, ratemepca, ratemegp, ratemeTrainX, ratemeTrainY, method='GP3')

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

def beautifyImFromPath(impath):
    im = cv2.imread(impath)

    landmarks = getLandmarks(im)
    compareMethods(im, landmarks, os.path.join(os.path.dirname(impath),"beautified.jpg"))

if __name__ == "__main__":
    GENDER = "F"

    dstFolder = "./results/"
    us10kdf = loadUS10KFacialFeatures()
    ratemedf = loadRateMeFacialFeatures()

    us10kgendered = us10kdf.loc[us10kdf['gender'] == GENDER]
    ratemegendered = ratemedf.loc[ratemedf['gender'] == GENDER]

    # split into training sets
    us10kTrainX = np.array(us10kgendered["facefeatures3D"].as_matrix().tolist())
    us10kTrainY = np.array(us10kgendered["attractiveness"].as_matrix().tolist())

    ratemeTrainX = np.array(ratemegendered["facefeatures3D"].as_matrix().tolist())
    ratemeTrainY = np.array(ratemegendered["attractiveness"].as_matrix().tolist())

    # load the GP that learnt attractiveness
    ratemepca, ratemegp = loadRateMePCAGP(type="3d", gender=GENDER)
    us10kpca, us10kgp = loadUS10KPCAGP(type="3d", gender=GENDER)

    print("begin beautification")

    im = cv2.imread("C:\\Users\\ellio\\Desktop\\test.png")
    compareMethods(im, ".\\beautified.jpg", us10kpca, ratemepca, us10kgp, ratemegp, us10kTrainX, us10kTrainY,
                   ratemeTrainX, ratemeTrainY)