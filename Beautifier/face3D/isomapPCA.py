import eos
import cv2
import numpy as np
import pickle
import os
import glob
from sklearn.decomposition import IncrementalPCA
from Beautifier.face3D.faceFeatures3D import model, blendshapes, createTextureMap

def processBatch(dstFolder, batch_num, batch_isomaps):
    print("saving batch {}".format(batch_num))
    isomaps = np.array(batch_isomaps)
    # calculate the sum from all pixels that have data
    meanIsomap = np.zeros((isomaps.shape[1], isomaps.shape[2], 4), dtype=np.float64)
    for y in range(isomaps.shape[1]):
        for x in range(isomaps.shape[2]):
            allPixels = isomaps[:, y, x, :]
            allPixels = allPixels[np.where(allPixels[:, 3] > 0)]

            if allPixels.shape[0] > 0:
                meanPixel = np.sum(allPixels, axis=0)
                meanPixel[3] = allPixels.shape[0]
            else:
                meanPixel = [0, 0, 0, 0]

            meanIsomap[y, x, :] = meanPixel

    # save out the batch and the summed isomap
    with open(os.path.join(dstFolder, "{}_isomaps.p".format(batch_num)), "wb") as file:
        pickle.dump(isomaps, file)
    with open(os.path.join(dstFolder, "{}_summean.p".format(batch_num)), "wb") as file:
        pickle.dump(meanIsomap, file)

def batchCreateIsomaps(dstFolder, GENDER, BATCH_SIZE = 50):
    from RateMe.RateMe import loadRateMeFacialFeatures

    dataframe = loadRateMeFacialFeatures()

    print("for each person create a isomap of their face")
    batch_isomaps = []
    batch_num = 0
    for index, row in dataframe.iterrows():
        print("{}/{}".format(index, len(dataframe)))

        gender = row["gender"]
        if gender != GENDER:
            continue

        facefeatures3D = row["facefeatures3D"]
        impaths = row["impaths"]
        poses = row["poses"]
        blendshape_coeffss = row["blendshape_coeffss"]

        # get a more accurate isomap for this  person
        avgFace = None
        for i in range(row["numUsableImages"]):
            impath = impaths[i]
            impath = impath.replace("E:\\Facedata\\RateMe", "C:\\FaceData\\RateMe")
            pose = poses[i]
            blendshape_coeffs = blendshape_coeffss[i]

            mesh = eos.morphablemodel.draw_sample(model, blendshapes, facefeatures3D, blendshape_coeffs, [])
            im = cv2.imread(impath)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
            isomap = createTextureMap(mesh, pose, im)

            isomap[:, :, 3] = isomap[:, :, 3] / 255

            # mirror the details on the otherside of the face if we have no data there
            isomap_f = cv2.flip(isomap, 1)
            hasInfo = np.where((isomap[:, :, 3] == 0) & (isomap_f[:, :, 3] == 1))
            isomap[hasInfo[0], hasInfo[1]] = isomap_f[hasInfo[0], hasInfo[1]]

            # cv2.imshow("im", cv2.cvtColor(im, cv2.COLOR_LAB2BGR))
            # cv2.imshow("isomap", cv2.cvtColor(isomap[:,:,:3].astype(np.uint8), cv2.COLOR_LAB2BGR))
            # cv2.waitKey(1)


            if avgFace is None:
                avgFace = isomap.copy().astype(np.float64)
            else:
                avgFace += isomap

        hasInfo = np.where(avgFace[:, :, 3] > 0)
        avgFace[hasInfo[0], hasInfo[1]] = avgFace[hasInfo[0], hasInfo[1]] / np.repeat(
            avgFace[hasInfo[0], hasInfo[1], 3][:, np.newaxis], 4, axis=1)

        # displayIm = avgFace.copy()
        # displayIm[:, :, 0] = isomap[:, :, 0]
        # displayIm = cv2.cvtColor(displayIm[:, :, :3].astype(np.uint8), cv2.COLOR_LAB2BGR)
        # cv2.imshow("avgface", displayIm.astype(np.uint8))

        # add this persons isomap to the datastructure
        batch_isomaps.append(avgFace)

        if len(batch_isomaps) == BATCH_SIZE:
            processBatch(dstFolder, batch_num, batch_isomaps)
            batch_num += 1
            batch_isomaps = []
    processBatch(dstFolder, batch_num, batch_isomaps)

def createMeanIsomap(dstFolder):
    meanIsomap = None
    for summean_path in glob.glob(os.path.join(dstFolder,"*_summean.p")):
        summean = pickle.load(open(summean_path, "rb"))

        if meanIsomap is None:
            meanIsomap = summean
        else:
            meanIsomap += summean

    hasData = np.where(meanIsomap[:, :, 3] > 0)
    meanIsomap[hasData[0], hasData[1], :3] = meanIsomap[hasData[0], hasData[1],:3] / np.repeat(
        meanIsomap[hasData[0], hasData[1], 3][:, np.newaxis], 3, axis=1)

    with open(os.path.join(dstFolder, "meanIsomap.p"), "wb") as file:
        pickle.dump(meanIsomap, file)

def fixMissingData(dstFolder):
    meanIsomap = pickle.load(open(os.path.join(dstFolder,"meanIsomap.p"), "rb"))

    for batch_isomaps_path in glob.glob(os.path.join(dstFolder, "*_isomaps.p")):
        batch_num = int(os.path.basename(batch_isomaps_path).split("_")[0])

        isomaps = pickle.load(open(batch_isomaps_path, "rb"))

        # add mean to the pixels that dont have data
        for y in range(isomaps.shape[1]):
            for x in range(isomaps.shape[2]):
                if meanIsomap[y,x,3] > 0:
                    missingData = np.where(isomaps[:, y, x, 3] == 0)
                    isomaps[missingData, y, x, :3] = meanIsomap[y,x,:3]

        #center the data
        isomaps[:, :, :, :3] -= meanIsomap[:, :, :3]

        with open(os.path.join(dstFolder, "{}_isomaps_centered.p".format(batch_num)), "wb") as file:
            pickle.dump(isomaps, file)

def trainPCA(dstFolder, BATCH_SIZE = 50):
    ipca = IncrementalPCA(n_components=10, batch_size=BATCH_SIZE)

    for batch_isomaps_path in glob.glob(os.path.join(dstFolder, "*_isomaps_centered.p")):
        batch_num = int(os.path.basename(batch_isomaps_path).split("_")[0])

        isomaps = pickle.load(open(batch_isomaps_path, "rb"))

        # remove the illumination and count channel
        isomaps_just_color = isomaps[:,:,:,1:3]

        #flatten to data
        X = isomaps_just_color.reshape(isomaps_just_color.shape[0], -1)  # ,isomaps_just_color.shape[3])

        ipca.partial_fit(X)

    with open(os.path.join(dstFolder, "pca.p"), "wb") as file:
        pickle.dump(ipca, file)

def getIsoMapFromPCA(pca, meanIsomap, X):
    XasImage = pca.inverse_transform(X)
    XasImage = XasImage.reshape(meanIsomap.shape[0], meanIsomap.shape[1], 2)
    mean_iso = meanIsomap[:, :, :3].copy()
    mean_iso[:, :, 1:] += XasImage

    return mean_iso

def visualisePCA(dstFolder):
    meanIsomap = pickle.load(open(os.path.join(dstFolder, "meanIsomap.p"), "rb"))
    pca = pickle.load(open(os.path.join(dstFolder, "pca.p"), "rb"))

    for i in range(10):
        for j in np.linspace(-1,1,3):
            X = [0,]*10
            X[i]=j*np.sqrt(pca.explained_variance_[i])
            displayIm = getIsoMapFromPCA(pca, meanIsomap, X)
            displayIm = cv2.cvtColor(displayIm[:, :, :3].astype(np.uint8), cv2.COLOR_LAB2BGR)
            cv2.imshow("pca_{}".format(j), displayIm.astype(np.uint8))
        # cv2.waitKey(-1)

    isomaps = pickle.load(open(os.path.join(dstFolder, "0_isomaps_centered.p"), "rb"))
    #play with variance
    for i in range(isomaps.shape[0]):
        isomap_just_color = isomaps[i, :, :, 1:3]

        # flatten to data
        X = isomap_just_color.reshape(1, -1)
        isomap_pca_color = pca.transform(X)[0]

        scale = 1
        # for j in range(5):
        new_isomap_pca_color = isomap_pca_color.copy()
        # new_isomap_pca_color[j] += np.sqrt(pca.explained_variance_[j]) * scale

        isomap_new_color = pca.inverse_transform(new_isomap_pca_color)
        isomap_new_color = isomap_new_color.reshape(isomap_just_color.shape[0],isomap_just_color.shape[1],isomap_just_color.shape[2])

        isomap_full = isomaps[i,:,:,:3].copy()
        isomap_full[:,:,1:] = isomap_new_color

        displayIm = isomaps[i, :, :, :3].copy() + meanIsomap[:, :, :3]
        displayIm = cv2.cvtColor(displayIm[:, :, :3].astype(np.uint8), cv2.COLOR_LAB2BGR)
        cv2.imshow("isomap_orig", displayIm.astype(np.uint8))

        displayIm = isomap_full.copy() + meanIsomap[:, :, :3]
        displayIm = cv2.cvtColor(displayIm[:, :, :3].astype(np.uint8), cv2.COLOR_LAB2BGR)
        cv2.imshow("isomap_new", displayIm.astype(np.uint8))
        cv2.waitKey(-1)


if __name__ == "__main__":

    GENDER = "F"
    dstFolder = "./resultsPCA/{}/".format(GENDER)

    # batchCreateIsomaps(dstFolder, GENDER)
    # createMeanIsomap(dstFolder)
    # fixMissingData(dstFolder)
    #
    # trainPCA(dstFolder)

    visualisePCA(dstFolder)