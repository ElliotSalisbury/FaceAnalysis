import os
import csv
import pandas as pd
import cv2
import numpy as np
from RateMe import ensureImageLessThanMax
from faceFeatures import getFaceFeatures
from beautifier import findBestFeaturesKNN,calculateLandmarksfromFeatures
from warpFace import warpFace
from face3D.faceFeatures3D import model,createTextureMap, getMeshFromLandmarks, exportMeshToJSON
from multiprocessing import Process

RateMeFolder = "E:\\Facedata\\RateMe"
combinedPath = os.path.join(RateMeFolder, "combined.csv")

def combineRatingCsvs():
    submissionFolders = [x[0] for x in os.walk(RateMeFolder)]

    #compile into single file
    with open(combinedPath, 'w', newline='') as wf:
        writer = csv.writer(wf)
        writer.writerow(("Folder", "Submission Title", "Submission Age", "Submission Gender", "Submission Author", "Rating Author","Rating", "Decimal", "Rating Text"))

        for folder in submissionFolders:
            ratingsPath = os.path.join(folder, "ratings.csv")

            if not os.path.exists(ratingsPath):
                continue

            with open(ratingsPath, 'r') as rf:
                reader = csv.reader(rf)

                for i, row in enumerate(reader):
                    if i==0:
                        continue
                    writer.writerow([folder,] + row)

# #load the data, and drop the rating text we wont need it
# df = pd.read_csv(combinedPath)
# df.drop('Rating Text', 1)
#
# meanRatings = df["Rating"].groupby(df["Submission Gender"]).mean()
# print(meanRatings)

def findBestFeaturesKNN(myFeatures, trainX, trainY, beautyAim):
    print("finding optimal face features KNN")
    # calculate nearest beauty weighted distance to neighbours
    weightedDistances = np.zeros((len(trainX), 1))
    for i in range(len(trainX)):
        neighborFeatures = trainX[i]
        neighborBeauty = 10 - abs(trainY[i]-beautyAim)
        distanceToNeighbor = np.linalg.norm(myFeatures - neighborFeatures)

        weightedDistance = neighborBeauty / distanceToNeighbor
        weightedDistances[i] = weightedDistance

    nearestWeightsIndexs = np.argsort(weightedDistances, 0)[::-1]

    # find the optimal K size for nearest neighbor
    K = 5
    kNewFeatures = np.zeros((K, len(myFeatures)))
    for k in range(K):
        indexs = nearestWeightsIndexs[:k + 1]
        weights = weightedDistances[indexs]
        features = trainX[indexs]
        kNewFeatures[k, :] = np.sum((weights * features), axis=0) / np.sum(weights)

    # y_pred = gp.predict(kNewFeatures)
    # bestK = np.argmax(y_pred, 0)

    return kNewFeatures[K-1]

def averageFaces(df, outpath):
    startIm = cv2.imread("E:\\Facedata\\10k US Adult Faces Database\\Face Images\\Aaron_Nickell_11_oval.jpg")
    startIm = ensureImageLessThanMax(startIm, maxsize=256)

    numFaces = 5
    minAttractiveness = df['attractiveness'].min()
    maxAttractiveness = df['attractiveness'].max()
    attractiveRange = maxAttractiveness - minAttractiveness
    attractiveWidth = attractiveRange / float(numFaces)
    halfAW = attractiveWidth / 2
    hotnessRange = np.linspace(minAttractiveness+halfAW,maxAttractiveness-halfAW,numFaces)

    startLandmarks, startFaceFeatures = getFaceFeatures(startIm)

    imfaces = np.zeros((startIm.shape[0]*2, startIm.shape[1]*len(hotnessRange), startIm.shape[2]), dtype=np.float32)

    grouped = df.groupby("gender")
    for i, (gender, group) in enumerate(grouped):
        for j, hotness in enumerate(hotnessRange):
            hotgroup = group.loc[group['attractiveness'] >= hotness-halfAW]
            hotgroup = hotgroup.loc[hotgroup['attractiveness'] < hotness+halfAW]

            if hotgroup.size == 0:
                continue

            hotImpaths = np.array(hotgroup["impath"].as_matrix().tolist())
            hotlandmarks = np.array(hotgroup["landmarks"].as_matrix().tolist())
            hotFacefeatures = np.array(hotgroup["facefeatures"].as_matrix().tolist())
            hotnessScore = np.array(hotgroup["attractiveness"].as_matrix().tolist())

            print("%s %d %d"%(gender,hotness,hotFacefeatures.shape[0]))

            hotFacefeatures = findBestFeaturesKNN(startFaceFeatures, hotFacefeatures, hotnessScore, hotness)

            newLandmarksKNN = calculateLandmarksfromFeatures(startLandmarks, hotFacefeatures)

            count = 0
            for k, impath in enumerate(hotImpaths):
                if k>300:
                    break

                im = cv2.imread(impath)
                im = ensureImageLessThanMax(im)
                imLandmarks = hotlandmarks[k]
                hotFace = warpFace(im, imLandmarks, newLandmarksKNN, justFace=True)

                #crop image to right size
                hotFace = hotFace[:startIm.shape[0],:startIm.shape[1]]

                # cv2.imshow("1face", hotFace)
                # cv2.waitKey(1)
                print("%s/%s"%(k,hotImpaths.shape[0]))

                count += 1
                imfaces[startIm.shape[0]*i:startIm.shape[0]*(i+1), startIm.shape[1]*j:startIm.shape[1]*(j+1)][:hotFace.shape[0],:hotFace.shape[1]] += hotFace
            imfaces[startIm.shape[0] * i:startIm.shape[0] * (i + 1), startIm.shape[1] * j:startIm.shape[1] * (j + 1)] /= count

            avgFace = np.uint8(imfaces[startIm.shape[0] * i:startIm.shape[0] * (i + 1), startIm.shape[1] * j:startIm.shape[1] * (j + 1)])
            cv2.imwrite(os.path.join(outpath,"averageFaces_%s_%0.2f_%d.jpg" % (gender, hotness, hotImpaths.shape[0])), avgFace)

            cv2.imshow("avg", np.uint8(imfaces))
            cv2.waitKey(1)

    imfaces = np.uint8(imfaces)
    cv2.imshow("sdfs", imfaces)
    cv2.imwrite(os.path.join(outpath,"averageFaces.jpg"), imfaces)
    cv2.waitKey(-1)

def averageFaces3D(df, outpath):
    numFaces = 5
    minAttractiveness = df['attractiveness'].min()
    maxAttractiveness = df['attractiveness'].max()
    attractiveRange = maxAttractiveness - minAttractiveness
    attractiveWidth = attractiveRange / float(numFaces)
    halfAW = attractiveWidth / 2
    hotnessRange = np.linspace(minAttractiveness+halfAW,maxAttractiveness-halfAW,numFaces)

    grouped = df.groupby("gender")
    for i, (gender, group) in enumerate(grouped):
        for j, hotness in enumerate(hotnessRange):
            hotgroup = group.loc[group['attractiveness'] >= hotness-halfAW]
            hotgroup = hotgroup.loc[hotgroup['attractiveness'] < hotness+halfAW]

            if hotgroup.size == 0:
                continue

            hotImpaths = np.array(hotgroup["impath"].as_matrix().tolist())
            hotlandmarks = np.array(hotgroup["landmarks"].as_matrix().tolist())
            hotFacefeatures = np.array(hotgroup["facefeatures3D"].as_matrix().tolist())
            hotnessScore = np.array(hotgroup["attractiveness"].as_matrix().tolist())

            Process(target=exportAverageFace, args=(outpath, gender, hotness, hotFacefeatures, hotImpaths, hotlandmarks)).start()

def exportAverageFace(outpath, gender, hotness, hotFacefeatures, hotImpaths, hotlandmarks):
    print("%s %d %d" % (gender, hotness, hotFacefeatures.shape[0]))

    avgHotFaceFeatures = hotFacefeatures.mean(axis=0)

    hotMeshVerts = model.get_shape_model().draw_sample(avgHotFaceFeatures[0:63]).reshape((-1, 3))
    hotMeshVerts2 = None

    count = 0
    avgFace = None
    for k, impath in enumerate(hotImpaths):
        # if k>5:
        #     break

        im = cv2.imread(impath)
        landmarks = hotlandmarks[k]
        mesh, pose, shape_coeffs, blendshape_coeffs = getMeshFromLandmarks(landmarks, im)
        isomap = createTextureMap(mesh, pose, im)
        isomap[:, :, 3] = isomap[:, :, 3] / 255
        if avgFace is None:
            avgFace = isomap.copy().astype(np.uint64)
        else:
            avgFace += isomap

        if hotMeshVerts2 is None:
            hotMeshVerts2 = np.array(mesh.vertices)
        else:
            hotMeshVerts2 += np.array(mesh.vertices)
        count += 1

        # countDivisor = np.repeat(avgFace[:,:,3][:,:,np.newaxis],3, axis=2)
        # cv2.imshow("face", (avgFace[:,:,:3]/countDivisor).astype(np.uint8))
        # cv2.waitKey(1)
        print("%s/%s" % (k, hotImpaths.shape[0]))

    countDivisor = np.repeat(avgFace[:, :, 3][:, :, np.newaxis], 3, axis=2)
    avgFace = avgFace[:, :, :3] / countDivisor

    cv2.imwrite(os.path.join(outpath, "averageFaces_%s_%0.2f_%d.jpg" % (gender, hotness, hotImpaths.shape[0])),
                avgFace)

    exportMeshToJSON(mesh, os.path.join(outpath,
                                        "averageFaces_%s_%0.2f_%d.json" % (gender, hotness, hotImpaths.shape[0])),
                     verts=hotMeshVerts[:, 0:3].flatten().tolist())
    hotMeshVerts2 = hotMeshVerts2 / count
    exportMeshToJSON(mesh, os.path.join(outpath,
                                        "averageFaces_%s_%0.2f_%d_2.json" % (gender, hotness, hotImpaths.shape[0])),
                     verts=hotMeshVerts2[:, 0:3].flatten().tolist())

if __name__ == "__main__":
    #load in the dataframes for analysis
    df = pd.read_pickle("../US10K/US10KData.p")

    averageFaces3D(df, "./us10k3D/")