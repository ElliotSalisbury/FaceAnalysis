import csv
import os
import pickle
import cv2
import dlib
import numpy as np
import pandas as pd
from Beautifier.faceFeatures import getFaceFeatures
from Beautifier.face3D.faceFeatures3D import getMeshFromLandmarks
from Beautifier.gaussianProcess import trainGP

#quickly change gender settings
GENDER_DICT = {0:"F",1:"M"}

#US10k data location
demographicscsv = "E:\\Facedata\\10k US Adult Faces Database\\Full Attribute Scores\\demographic & others labels\\demographic-others-labels-final.csv"
imfolder = "E:\\Facedata\\10k US Adult Faces Database\\Face Images"
#output location
scriptFolder = os.path.dirname(os.path.realpath(__file__))


def readUS10kDemographics():
    print("reading US10k demographics data")
    demographicsData = []
    with open(demographicscsv, 'r') as demoF:
        reader = csv.reader(demoF)
        header = []
        for i, row in enumerate(reader):
            if i == 0:
                header = row
                continue

            row[0] = os.path.join(imfolder, row[0])
            rowDict = {header[j]:row[j] for j in range(len(row)) if os.path.isfile(row[0])}
            demographicsData.append(rowDict)
    return demographicsData

def saveFacialFeatures(demographicsData):
    print("calculating face features")
    allData = []
    for i, data in enumerate(demographicsData):
        attractiveScore = float(data['Attractive'])
        gender = GENDER_DICT[int(data['Gender'])]
        if not np.isnan(attractiveScore):
            impath = data['Filename']
            im = cv2.imread(impath)

            try:
                landmarks, faceFeatures = getFaceFeatures(im)
                mesh, pose, shape_coeffs, blendshape_coeffs = getMeshFromLandmarks(landmarks, im, num_shape_coefficients_to_fit=10)

                dataDict = {"gender": gender, "attractiveness": attractiveScore,
                            "landmarks": landmarks, "facefeatures": faceFeatures,
                            "facefeatures3D": shape_coeffs, "mesh": mesh, "pose": pose, "blendshape_coeffs": blendshape_coeffs,
                            "impath": impath
                            }

                allData.append(dataDict)
                print("%i / %i"%(i,len(demographicsData)))
            except:
                continue

    allDataDF = pd.DataFrame(allData)
    allDataDF.to_pickle(os.path.join(scriptFolder,"US10kData.p"))

    return allDataDF

def loadUS10kFacialFeatures():
    return pd.read_pickle(os.path.join(scriptFolder, "US10kData.p"))
def loadUS10kPCAGP(type="2d", gender="F"):
    return pickle.load(open(os.path.join(scriptFolder, "%s/GP_%s.p"%(type,gender)), "rb"))
def loadUS10k(type="2d", gender="F"):
    df = loadUS10kFacialFeatures()

    df_G = df.loc[df['gender'] == gender]

    featuresIndex = "facefeatures"
    if type == "3d":
        featuresIndex = "facefeatures3D"

    trainX = np.array(df_G[featuresIndex].as_matrix().tolist())
    trainY = np.array(df_G["attractiveness"].as_matrix().tolist())
    pca, gp = loadUS10kPCAGP(type=type, gender=gender)

    return trainX, trainY, pca, gp

if __name__ == "__main__":
    demographicsData = readUS10kDemographics()
    df = saveFacialFeatures(demographicsData)
    # df = loadUS10kFacialFeatures()

    trainGP(df, os.path.join(scriptFolder, "2d"), trainPercentage=0.8)
    trainGP(df, os.path.join(scriptFolder, "3d"), trainPercentage=0.8, featureset="facefeatures3D", train_on_PCA=False)