import glob
import os
import cv2
import pandas as pd
import pickle
from Beautifier.gaussianProcess import trainGP
from Beautifier.faceFeatures import getFaceFeatures
from Beautifier.face3D.faceFeatures3D import getMeshFromMultiLandmarks
import numpy as np

MAX_IM_SIZE = 512

def ensureImageLessThanMax(im, maxsize=MAX_IM_SIZE):
    height, width, depth = im.shape
    if height > maxsize or width > maxsize:

        if width > height:
            ratio = maxsize / float(width)
            width = maxsize
            height = int(height * ratio)
        else:
            ratio = maxsize / float(height)
            height = maxsize
            width = int(width * ratio)
        im = cv2.resize(im,(width,height))
    return im

scriptFolder = os.path.dirname(os.path.realpath(__file__))
def saveFacialFeatures(combinedcsvpath):
    df = pd.read_csv(combinedcsvpath)
    df.drop('Rating Text', 1)
    df.drop('Submission Title', 1)

    #filter out the weird ages
    filtered = df.loc[df['Submission Age'] >= 18]
    filtered = filtered.loc[filtered['Submission Age'] < 50]

    grouped = filtered.groupby(['Submission Gender', "Folder"])

    allData = []
    for i, (genderfolder, group) in enumerate(grouped):
        gender = genderfolder[0]
        folder = genderfolder[1]
        rating = group["Rating"].mean()

        #get the image files:
        types = ('*.jpg', '*.png', '*.bmp')
        impaths = []
        for type in types:
            impaths.extend(glob.glob(os.path.join(folder,type)))

        usedImPaths = []
        ims = []
        landmarkss = []
        faceFeaturess = []
        for impath in impaths:
            im = cv2.imread(impath)
            im = ensureImageLessThanMax(im)

            try:
                landmarks, faceFeatures = getFaceFeatures(im)

                usedImPaths.append(impath)
                ims.append(im)
                landmarkss.append(landmarks)
                faceFeaturess.append(faceFeatures)
            except:
                continue

        if len(ims) > 0:
            meshs, poses, shape_coeffs, blendshape_coeffss = getMeshFromMultiLandmarks(landmarkss, ims, num_shape_coefficients_to_fit=10)

            dataDict = {"gender": gender, "attractiveness": rating,
                        "impaths": usedImPaths, "numImages": len(ims),
                        "landmarkss": landmarkss, "facefeaturess": faceFeaturess,
                        "meshs": meshs, "poses": poses, "facefeatures3D": shape_coeffs, "blendshape_coeffss": blendshape_coeffss
                        }
            allData.append(dataDict)
            print("%i / %i" % (i, len(grouped)))

    allDataDF = pd.DataFrame(allData)
    allDataDF = allDataDF.sample(frac=1).reset_index(drop=True)
    allDataDF.to_pickle(os.path.join(scriptFolder,"RateMeData.p"))

    return allDataDF

def dataFrameTo2D(df):
    allData = []
    for index, row in df.iterrows():
        gender = row["gender"]
        attractiveness = row["attractiveness"]
        facefeatures3D = row["facefeatures3D"]

        impaths = row["impaths"]
        landmarkss = row["landmarkss"]
        facefeaturess = row["facefeaturess"]
        meshs = row["meshs"]
        poses = row["poses"]
        blendshape_coeffss = row["blendshape_coeffss"]

        for i in range(row["numImages"]):
            impath = impaths[i]
            landmarks = landmarkss[i]
            facefeatures = facefeaturess[i]
            mesh = meshs[i]
            pose = poses[i]
            blendshape_coeffs = blendshape_coeffss[i]

            dataDict = {"gender": gender, "attractiveness": attractiveness,
                        "landmarks": landmarks, "facefeatures": facefeatures,
                        "facefeatures3D": facefeatures3D, "mesh":mesh, "pose":pose, "blendshape_coeffs": blendshape_coeffs,
                        "impath": impath}
            allData.append(dataDict)
    allDataDF = pd.DataFrame(allData)
    allDataDF = allDataDF.sample(frac=1).reset_index(drop=True)
    return allDataDF

def loadRateMeFacialFeatures():
    return pd.read_pickle(os.path.join(scriptFolder, "RateMeData.p"))
def loadRateMePCAGP(type="2d", gender="F"):
    return pickle.load(open(os.path.join(scriptFolder, "%s/GP_%s.p"%(type,gender)), "rb"))
def loadRateMe(type="2d", gender="F"):
    df = loadRateMeFacialFeatures()
    if type=="2d":
        df = dataFrameTo2D(df)

    df_G = df.loc[df['gender'] == gender]

    featuresIndex = "facefeatures"
    if type=="3d":
        featuresIndex = "facefeatures3D"

    trainX = np.array(df_G[featuresIndex].as_matrix().tolist())
    trainY = np.array(df_G["attractiveness"].as_matrix().tolist())
    pca, gp = loadRateMePCAGP(type=type, gender=gender)

    return trainX, trainY, pca, gp

if __name__ == "__main__":
    # rateMeFolder = "E:\\Facedata\\RateMe"
    # combinedPath = os.path.join(rateMeFolder, "combined.csv")
    #
    # df = saveFacialFeatures(combinedPath)
    df = loadRateMeFacialFeatures()

    df2d = dataFrameTo2D(df)
    trainGP(df2d, os.path.join(scriptFolder, "2d"), trainPercentage=0.9)

    moreaccurate = df[df["numImages"]>=3]
    trainGP(moreaccurate, os.path.join(scriptFolder, "3d"), trainPercentage=0.9, featureset="facefeatures3D", train_on_PCA=False)