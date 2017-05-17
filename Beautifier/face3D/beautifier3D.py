import numpy as np
import cv2
from Beautifier.beautifier import findBestFeaturesKNN, LEFT_EYE_POINTS, RIGHT_EYE_POINTS, MOUTH_POINTS, NOSE_POINTS
from Beautifier.warpFace import warpFace, drawLandmarks
from Beautifier.faceFeatures import getLandmarks
from Beautifier.face3D.faceFeatures3D import getFaceFeatures3D, getMeshFromLandmarks, model, blendshapes, getFaceFeatures3D2DFromMesh, faceLines3D2D
from Beautifier.face3D.warpFace3D import warpFace3D, projectVertsTo2D
import eos
import os
import scipy

scriptFolder = os.path.dirname(os.path.realpath(__file__))

alphaWeighting = []
for line in faceLines3D2D:
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

def calculateFeatures3DfromFeatures3D2D(originalFeatures3D, optimalFaceFeatures3D2D):
    print("minimising cost of distance constraints from facial features")

    # cost function used to find the shape coefficients that match the 3D2D face features
    iterCount = 0
    def costFunction(features3D):
        nonlocal iterCount

        newMesh = eos.morphablemodel.draw_sample(model, blendshapes, features3D, [], [])  # newFaceFeatures[63:], [])

        verts, newFaceFeatures3D2D = getFaceFeatures3D2DFromMesh(newMesh)

        distanceSqError = np.sum(alphaWeighting * np.square(newFaceFeatures3D2D - optimalFaceFeatures3D2D))

        iterCount += 1
        # if iterCount % 100 == 0:
        print("%i - %0.2f" % (iterCount, distanceSqError))

        return distanceSqError

    # find facial landmarks that fit these new distances
    # newFeatures3D = scipy.optimize.minimize(costFunction, originalFeatures3D)
    newFeatures3D = scipy.optimize.minimize(costFunction, originalFeatures3D, method='SLSQP', options={"maxiter": 5, "eps": 0.001})
    newFeatures3D = newFeatures3D.x

    return newFeatures3D

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

def findBestFeaturesOptimisation3D2D(features3D, mesh, pca, gp):
    print("finding optimal face features optimisation")
    iterCount = 0

    def GPCostFunction(features3D):
        nonlocal iterCount
        newMesh = eos.morphablemodel.draw_sample(model, blendshapes, features3D, [], [])
        landmarks3D, features3D2D = getFaceFeatures3D2DFromMesh(newMesh)

        reducedFeatures = pca.transform([features3D2D])
        y_pred = gp.predict(reducedFeatures)

        LP = np.sum((-np.square(reducedFeatures)) / (2 * pca.explained_variance_))

        iterCount += 1
        # if iterCount % 100 == 0:
        print("%i - %0.2f - %0.2f"%(iterCount, y_pred, LP))

        alpha = 0.3
        return (alpha-1)*y_pred - alpha*LP

    bounds = np.zeros((features3D.shape[0],2))
    bounds[:, 0] = features3D - 2.15
    bounds[:,1] = features3D + 2.15

    optimalNewFaceFeatures = scipy.optimize.minimize(GPCostFunction, features3D, method='SLSQP', options={"maxiter":5,"eps":0.001})
    return optimalNewFaceFeatures.x

def beautifyFace3D(im, landmarks, pca, gp, trainX, trainY, method='KNN', exaggeration=0.5):
    mesh, pose, shape_coeffs, blendshape_coeffs = getMeshFromLandmarks(landmarks, im, num_shape_coefficients_to_fit=-1)
    features3D = np.array(shape_coeffs)

    if method=='KNN':
        newFeatures3D = findBestFeaturesKNN(features3D, pca, gp, trainX, trainY)
    elif method == 'GP':
        newFeatures3D = findBestFeaturesOptimisation(features3D, gp)

    delta = newFeatures3D - features3D
    newFeatures3D = features3D + (delta*exaggeration)

    newMesh = eos.morphablemodel.draw_sample(model, blendshapes, newFeatures3D, blendshape_coeffs,[])#newFaceFeatures[63:], [])

    warpedIm = warpFace3D(im, mesh, pose, newMesh)
    return warpedIm

def beautifyFace3D2D(im, landmarks, pca, gp, trainX, trainY, method='KNN', exaggeration=0.5):
    mesh, pose, shape_coeffs, blendshape_coeffs = getMeshFromLandmarks(landmarks, im, num_shape_coefficients_to_fit=-1, num_iterations=300)
    features3D = np.array(shape_coeffs)
    landmarks3D, features3D2D = getFaceFeatures3D2DFromMesh(mesh)

    # if method=='KNN':
    #     newFaceFeatures3D2D = findBestFeaturesKNN(features3D2D, pca, gp, trainX, trainY)
    # elif method == 'GP':
    #     newFaceFeatures3D2D = findBestFeaturesOptimisation3D2D(features3D2D, pca, gp)
    newFeatures3D = findBestFeaturesOptimisation3D2D(features3D, mesh, pca, gp)

    # # construct the landmarks that satisify the distance constraints of the features
    # newFeatures3D = calculateFeatures3DfromFeatures3D2D(features3D, newFaceFeatures3D2D)
    #
    # newMesh = eos.morphablemodel.draw_sample(model, blendshapes, newFeatures3D, blendshape_coeffs, [])  # newFaceFeatures[63:], [])
    #
    # warpedIm = warpFace3D(im, mesh, pose, newMesh)
    # return warpedIm

    delta = newFeatures3D - features3D
    newFeatures3D = features3D + (delta * exaggeration)

    newMesh = eos.morphablemodel.draw_sample(model, blendshapes, newFeatures3D, blendshape_coeffs, [])  # newFaceFeatures[63:], [])

    warpedIm = warpFace3D(im, mesh, pose, newMesh, accurate=True)
    return warpedIm

def compareMethods(im, datasets):
    landmarks = getLandmarks(im)

    displayIm = np.zeros((im.shape[0] * len(datasets), im.shape[1] * 4, im.shape[2]), dtype=np.uint8)
    displayIm[:im.shape[0], :im.shape[1], :] = im.copy()
    for i, dataset in enumerate(datasets):
        trainX, trainY, pca, gp = dataset

        # KNN = beautifyFace3D2D(im, landmarks, pca, gp, trainX, trainY, method='KNN', exaggeration=1)
        GP = beautifyFace3D2D(im, landmarks, pca, gp, trainX, trainY, method='GP', exaggeration=1)

        # displayIm[im.shape[0] * i:im.shape[0] * (i + 1), im.shape[1]:im.shape[1] * 2, :] = KNN
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

    dstFolder = "./results3d2d{}/".format(GENDER)

    datasets = [#loadUS10k(type="3d", gender=GENDER),
                loadRateMe(type="3d2d", gender=GENDER)]

    print("begin beautification")
    import glob

    # imFolder = r"C:\Users\ellio\PycharmProjects\circlelines\Beautifier\face3D\me\*.jpg"
    # imFolder = r"E:\Facedata\RateMe\21_F_423ekl\*.jpg" # 8.1
    imFolder = r"E:\Facedata\RateMe\21_F_42mxvl\*.jpg" # 6.5
    # imFolder = r"E:\Facedata\10k US Adult Faces Database\Publication Friendly 49-Face Database\49 Face Images\*.jpg"
    for i, impath in enumerate(glob.glob(imFolder)):
        print(i)
        try:
            im = cv2.imread(impath)
            filename = os.path.basename(impath)

            # from Beautifier.face3D.warpFace3D import drawMesh
            # landmarks = getLandmarks(im)
            # mesh, pose, shape_coeffs, blendshape_coeffs = getMeshFromLandmarks(landmarks, im, num_shape_coefficients_to_fit=-1, num_iterations=300)
            # meshim = drawMesh(im,mesh,pose)

            # cv2.imshow("mesh", meshim)
            # cv2.waitKey(-1)

            comparedIm = compareMethods(im, datasets)

            cv2.imshow("face", comparedIm)
            cv2.waitKey(1)
            cv2.imwrite(os.path.join(dstFolder, filename), comparedIm)

            compareIndex = 0
            cv2.imwrite(os.path.join(dstFolder, "compare/{}_1.jpg".format(filename)), im)
            cv2.imwrite(os.path.join(dstFolder, "compare/{}_2.jpg".format(filename)), comparedIm[im.shape[0] * compareIndex:im.shape[0] * (compareIndex + 1), im.shape[1] * 2:im.shape[1] * 3, :])
        except:
            continue