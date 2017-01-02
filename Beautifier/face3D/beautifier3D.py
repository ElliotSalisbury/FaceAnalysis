import numpy as np
import cv2
import pickle
import scipy
from sklearn import gaussian_process
from warpFace import warpFace
from US10K import loadUS10KFacialFeatures
from RateMe import loadRateMeFacialFeatures
from beautifier import findBestFeaturesKNN, findBestFeaturesOptimisation, findBestFeaturesOptimisation2, findBestFeaturesOptimisation3
from faceFeatures import getFaceFeatures
from face3D.faceFeatures3D import getFaceFeatures3D, getMeshFromLandmarks, model, blendshapes
from face3D.warpFace3D import warpFace3D
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
    newMesh = eos.morphablemodel.draw_sample(model, blendshapes, newFaceFeatures[:63], newFaceFeatures[63:], [])

    warpedIm = warpFace3D(im, mesh, pose, newMesh)
    return warpedIm

def compareMethods(im, landmarks, outpath):
    US10KKNN = beautifyFace3D(im, landmarks, us10kpca, us10kgp, trainX, trainY, method='KNN')
    US10KGP = beautifyFace3D(im, landmarks, us10kpca, us10kgp, trainX, trainY, method='GP3')
    # RateMeKNN = beautifyFace3D(im, landmarks, features, ratemepca, ratemegp, trainXRateMe, trainYRateMe, method='KNN')
    # RateMeGP = beautifyFace3D(im, landmarks, features, ratemepca, ratemegp, trainXRateMe, trainYRateMe, method='GP3')

    displayIm = np.zeros((im.shape[0] * 2, im.shape[1] * 4, im.shape[2]), dtype=np.uint8)
    displayIm[:im.shape[0], :im.shape[1], :] = im.copy()
    displayIm[:im.shape[0], im.shape[1]:im.shape[1] * 2, :] = US10KKNN
    displayIm[:im.shape[0], im.shape[1] * 2:im.shape[1] * 3, :] = US10KGP
    # displayIm[im.shape[0]:, im.shape[1]:im.shape[1] * 2, :] = RateMeKNN
    # displayIm[im.shape[0]:, im.shape[1] * 2:im.shape[1] * 3, :] = RateMeGP

    diff = np.abs(np.float32(im) - np.float32(US10KKNN))
    diff = (diff / np.max(diff)) * 255
    displayIm[:im.shape[0], im.shape[1] * 3:im.shape[1] * 4, :] = np.uint8(diff)

    # diff = np.abs(np.float32(im) - np.float32(RateMeGP))
    # diff = (diff / np.max(diff)) * 255
    # displayIm[im.shape[0]:, im.shape[1] * 3:im.shape[1] * 4, :] = np.uint8(diff)

    cv2.imshow("face", displayIm)
    cv2.waitKey(-1)
    cv2.imwrite(outpath, displayIm)

def beautifyImFromPath(impath):
    im = cv2.imread(impath)

    landmarks, faceFeatures = getFaceFeatures(im)
    compareMethods(im, landmarks, os.path.join(os.path.dirname(impath),"beautified.jpg"))

if __name__ == "__main__":
    dstFolder = "./results/"
    us10kdf = loadUS10KFacialFeatures()
    # ratemedf = loadRateMeFacialFeatures()

    us10kgendered = us10kdf.loc[us10kdf['gender'] == 'M']
    us10kgendered = us10kgendered.loc[us10kgendered['attractiveness'] >= 4]
    # ratemewomen = ratemedf.loc[ratemedf['gender'] == 'F']

    #split into training sets
    trainSize = int(us10kgendered.shape[0] * 0.8)
    traindf = us10kgendered[:trainSize]
    trainX = np.array(traindf["facefeatures3D"].as_matrix().tolist())
    trainY = np.array(traindf["attractiveness"].as_matrix().tolist())

    testdf = us10kgendered[trainSize:]
    testX = np.array(testdf["facefeatures3D"].as_matrix().tolist())
    testY = np.array(testdf["attractiveness"].as_matrix().tolist())
    # testL = np.array(testdf["landmarks"].as_matrix().tolist())
    # testI = np.array(testdf["impath"].as_matrix().tolist())

    # trainXRateMe = np.array(ratemewomen["facefeatures"].as_matrix().tolist())
    # trainYRateMe = np.array(ratemewomen["attractiveness"].as_matrix().tolist())

    #load the GP that learnt attractiveness
    # ratemepca, ratemegp = pickle.load(
    #     open(os.path.join(scriptFolder,"../rRateMe/3d/GP_F.p"), "rb"))
    us10kpca, us10kgp = pickle.load(
        open(os.path.join(scriptFolder,"../../US10k/3d/GP_M.p"), "rb"))

    print("begin beautification")

    beautifyImFromPath("C:\\Users\\Elliot\\Desktop\\tarin\\IMG_1727.JPG")