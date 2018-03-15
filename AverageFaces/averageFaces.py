import os
import csv
import cv2
import numpy as np
from RateMe.RateMe import ensureImageLessThanMax, loadRateMeFacialFeatures
from Beautifier.faceFeatures import getFaceFeatures
from Beautifier.beautifier import findBestFeaturesKNN,calculateLandmarksfromFeatures
from Beautifier.warpFace import warpFace
from Beautifier.face3D.faceFeatures3D import BFM_FACEFITTING

import random
import msgpack
import dlib

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

    # df = df[df["numImages"] >= 2]

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

            hotFacefeatures = np.array(hotgroup["facefeaturesCNN"].as_matrix().tolist())
            avgHotFaceFeatures = hotFacefeatures.mean(axis=0)

            title = "averageFaces_%s_%0.2f_%d" % (gender, hotness, hotFacefeatures.shape[0])
            # title = "averageFaces_%s_%d" % (gender, j)
            hotMesh = BFM_FACEFITTING.getMeshFromShapeCeoffs(avgHotFaceFeatures)

            # with open(os.path.join(outpath, title+".msg"), 'wb') as outfile:
            #     meshjson = exportMeshToJSON(hotMesh)
            #     msgpack.dump(meshjson,outfile)
            #     json.dump(meshjson, outfile, indent=4, sort_keys=True)


            #create the texture from the faces
            NUMSAMPLES = min(len(hotFacefeatures), 300)
            randomSampleIndexs = random.sample(range(len(hotFacefeatures)), NUMSAMPLES)

            hotAttractiveness = np.array(hotgroup["attractiveness"].as_matrix().tolist())
            hotImpathss = np.array(hotgroup["impaths"].as_matrix().tolist())
            hotLandmarkss = np.array(hotgroup["landmarkss"].as_matrix().tolist())
            hotPosess = np.array(hotgroup["poses"].as_matrix().tolist())
            hotBlendshape_coeffss = np.array(hotgroup["blendshape_coeffss"].as_matrix().tolist())
            avgFace=None

            # allim_ws = []
            # allim_hs = []
            # alllandmarks = []
            # for k in randomSampleIndexs:
            #     impaths = hotImpathss[k]
            #     randomImIndexs = random.sample(range(len(impaths)), 1)
            #
            #     for index in randomImIndexs:
            #         im = cv2.imread(impaths[index])
            #         allim_ws.append(im.shape[1])
            #         allim_hs.append(im.shape[0])
            #     landmarks = [hotLandmarkss[k][index] for index in randomImIndexs]
            #     alllandmarks.extend(landmarks)
            #
            # randomImIndexs = random.sample(range(len(allim_ws)), min(len(allim_ws), 50))
            # allim_ws = [allim_ws[index] for index in randomImIndexs]
            # allim_hs = [allim_hs[index] for index in randomImIndexs]
            # alllandmarks = [alllandmarks[index] for index in randomImIndexs]
            # _, _, shape, _ = getMeshFromMultiLandmarks_IWH(alllandmarks, allim_ws, allim_hs, num_shape_coefficients_to_fit=10)
            # hotMesh = eos.morphablemodel.draw_sample(model, blendshapes, shape, [], [])
            #
            # with open(os.path.join(outpath, title + "_fitted.msg"), 'w') as outfile:
            #     meshjson = exportMeshToJSON(hotMesh)
            #     msgpack.dump(meshjson, outfile)
            #     # json.dump(meshjson, outfile, indent=4, sort_keys=True)

            # for k in randomSampleIndexs:
            #     attractiveness = hotAttractiveness[k]
            #     impaths = hotImpathss[k]
            #     poses = hotPosess[k]
            #     faceFeatures = hotFacefeatures[k]
            #     blendshapes_coeffs = hotBlendshape_coeffss[k]
            #
            #     yaws = np.array([pose.get_rotation_euler_angles()[1] for pose in poses])
            #     frontalIndex = np.argmin(np.absolute(yaws))
            #
            #     impath = impaths[frontalIndex]
            #     im = cv2.imread(impath)
            #     # im = ensureImageLessThanMax(im)
            #
            #     mesh = eos.morphablemodel.draw_sample(model, blendshapes, faceFeatures, blendshapes_coeffs[frontalIndex], [])
            #     pose = poses[frontalIndex]
            #
            #     isomap = createTextureMap(mesh, pose, im)
            #
            #     foldername = impath.split("\\")[-2]
            #     isoout = os.path.join(outpath, "faces/%0.2f_%s.png" % (attractiveness,foldername))
            #     cv2.imwrite(isoout, isomap)
            #
            #     isomap[:, :, 3] = isomap[:, :, 3] / 255
            #     if avgFace is None:
            #         avgFace = isomap.copy().astype(np.uint64)
            #     else:
            #         avgFace += isomap
            #
            #     print("%s %d %d %s/%s" % (gender, hotness, hotFacefeatures.shape[0], k, hotFacefeatures.shape[0]))
            # countDivisor = np.repeat(avgFace[:, :, 3][:, :, np.newaxis], 3, axis=2)
            # avgFace = avgFace[:, :, :3] / countDivisor
            #
            # cv2.imwrite(os.path.join(outpath, title+".jpg"),avgFace)

def ensureImageSmallestDimension(im, dimsize=40):
    height, width, depth = im.shape
    if width > height:
        ratio = dimsize / float(height)
        height = dimsize
        width = int(width * ratio)
    else:
        ratio = dimsize / float(width)
        width = dimsize
        height = int(height * ratio)
    if width < dimsize:
        width = dimsize
    if height < dimsize:
        height = dimsize
    im = cv2.resize(im,(width,height))
    return im

def averageFacesImage(df, outpath):
    detector = dlib.get_frontal_face_detector()
    numRows = 7
    facesPerRow = 26
    imageSize = 30

    FULLIm = np.zeros((numRows * imageSize, facesPerRow * imageSize*2, 3), dtype=np.uint8)

    # df = df[df["numImages"] >= 2]
    df = df.sort(['attractiveness'])
    grouped = df.groupby("gender")
    for i, (gender, group) in enumerate(grouped):
        fullIm = np.zeros((numRows * imageSize, facesPerRow * imageSize, 3), dtype=np.uint8)

        numPeople = group.shape[0]
        numPeoplePerRow = int(numPeople / numRows)
        for j in range(numRows):
            rowPeople = group.iloc[numPeoplePerRow*j:numPeoplePerRow*(j+1)]

            # randomSampleIndexs = np.linspace(0,rowPeople.shape[0]-1,facesPerRow).astype(np.int32)
            randomSampleIndexs = random.sample(range(numPeoplePerRow), facesPerRow)

            impathss = np.array(rowPeople["impaths"].as_matrix().tolist())
            attractiveness = np.array(rowPeople["attractiveness"].as_matrix().tolist())
            for l,k in enumerate(randomSampleIndexs):
                impaths = impathss[k]
                randomImIndexs = random.sample(range(len(impaths)), len(impaths))

                for index in randomImIndexs:
                    im = cv2.imread(impaths[index])

                    rects = detector(im, 1)
                    if len(rects) != 1:
                        continue
                    rect = rects[0]
                    ryt = max(rect.top() - (rect.height()*0.2)      ,0)
                    ryb = min(rect.bottom() + (rect.height()*0.2)   , im.shape[0])
                    rxl = max(rect.left() - (rect.width()*0.2)      ,0)
                    rxr = min(rect.right() + (rect.width()*0.2)     ,im.shape[1])

                    faceim = im[ryt:ryb, rxl:rxr]
                    if faceim.shape[0] < 40 or faceim.shape[1] < 40:
                        continue
                    faceim = ensureImageSmallestDimension(faceim, imageSize)

                    y=j*imageSize
                    x=l*imageSize
                    print("%d,%d"%(x,y))

                    fullIm[y:y + imageSize, x:x + imageSize, :] = faceim[:imageSize, :imageSize, :]

                    x = (l*(imageSize*2)) + (imageSize*((i+j)%2))
                    FULLIm[y:y + imageSize, x:x + imageSize, :] = faceim[:imageSize, :imageSize, :]
                    cv2.imshow("faces", FULLIm)
                    cv2.waitKey(1)
                    break

        cv2.imwrite(os.path.join(outpath, "allfaces_%s.jpg"%gender),fullIm)
    cv2.imwrite(os.path.join(outpath, "allfaces.jpg" ), FULLIm)


if __name__ == "__main__":
    #load in the dataframes for analysis
    df = loadRateMeFacialFeatures()

    averageFaces3D(df, "./rateme3D/")
    # averageFacesImage(df, "./rateme3D/")