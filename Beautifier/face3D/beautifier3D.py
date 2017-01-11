import numpy as np
import cv2
from US10k.US10k import loadUS10kFacialFeatures, loadUS10kPCAGP
from RateMe.RateMe import loadRateMeFacialFeatures, loadRateMePCAGP
from Beautifier.beautifier import findBestFeaturesKNN, findBestFeaturesOptimisation, findBestFeaturesOptimisation2, findBestFeaturesOptimisation3
from Beautifier.faceFeatures import getLandmarks
from Beautifier.face3D.faceFeatures3D import getFaceFeatures3D, getMeshFromLandmarks, model, blendshapes
from Beautifier.face3D.warpFace3D import warpFace3D
import eos
import os

scriptFolder = os.path.dirname(os.path.realpath(__file__))

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

def beautifyImFromPath(impath):
    im = cv2.imread(impath)

    landmarks = getLandmarks(im)
    compareMethods(im, landmarks, os.path.join(os.path.dirname(impath),"beautified.jpg"))

if __name__ == "__main__":
    import glob
    from Beautifier.face3D.faceFeatures3D import ensureImageLessThanMax
    from Beautifier.beautifier import beautifyIm

    GENDER = "F"

    dstFolder = "./results/"
    us10kdf = None#loadUS10kFacialFeatures()
    ratemedf = loadRateMeFacialFeatures()

    us10kgendered = None#us10kdf.loc[us10kdf['gender'] == GENDER]
    ratemegendered = ratemedf.loc[ratemedf['gender'] == GENDER]
    ratemegendered = ratemegendered.loc[ratemegendered['attractiveness'] > 8]

    # split into training sets
    us10kTrainX = None#np.array(us10kgendered["facefeatures3D"].as_matrix().tolist())
    us10kTrainY = None#np.array(us10kgendered["attractiveness"].as_matrix().tolist())

    ratemeTrainX = np.array(ratemegendered["facefeatures3D"].as_matrix().tolist())
    ratemeTrainY = np.array(ratemegendered["attractiveness"].as_matrix().tolist())

    # load the GP that learnt attractiveness
    ratemepca, ratemegp = loadRateMePCAGP(type="3d", gender=GENDER)
    us10kpca, us10kgp = None,None#loadUS10kPCAGP(type="3d", gender=GENDER)

    im = cv2.imread("C:\\Users\\ellio\\Desktop\\front.jpg")
    im = ensureImageLessThanMax(im, maxsize=1024)
    landmarks = getLandmarks(im)
    better = beautifyFace3D(im, landmarks, ratemepca, ratemegp, ratemeTrainX, ratemeTrainY, method="KNN")

    cv2.imshow("b", better)
    cv2.imshow("o", im)
    cv2.waitKey(-1)

    # print("begin beautification")
    #
    # for impath in glob.glob("E:\\Facedata\\10k US Adult Faces Database\\Publication Friendly 49-Face Database\\49 Face Images\\*.jpg"):
    #     im = cv2.imread(impath)
    #     compareMethods(im, impath+".beautified.jpg", us10kpca, ratemepca, us10kgp, ratemegp, us10kTrainX, us10kTrainY,ratemeTrainX, ratemeTrainY)