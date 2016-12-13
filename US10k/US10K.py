import csv
import os

import cv2
import dlib
import numpy as np
import pandas as pd
from faceFeatures import getFaceFeatures
from gaussianProcess import trainGP

#quickly change gender settings
GENDER_DICT = {0:"F",1:"M"}

#US10K data location
demographicscsv = "E:\\Facedata\\10k US Adult Faces Database\\Full Attribute Scores\\demographic & others labels\\demographic-others-labels-final.csv"
imfolder = "E:\\Facedata\\10k US Adult Faces Database\\Face Images"
#output location
scriptFolder = os.path.realpath(__file__)


def readUS10kDemographics():
    print("reading US10K demographics data")
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

            landmarks, faceFeatures = getFaceFeatures(im)

            if faceFeatures is not None:
                dataDict = {"gender": gender, "attractiveness": attractiveScore, "landmarks": landmarks,
                            "facefeatures": faceFeatures,
                            "impath": impath}
            allData.append(dataDict)

    allDataDF = pd.DataFrame(allData)
    allDataDF.to_pickle(os.path.join(scriptFolder,"US10KData.p"))

    return allDataDF

def loadUS10KFacialFeatures():
    return pd.read_pickle(os.path.join(scriptFolder, "US10KData.p"))

if __name__ == "__main__":
    demographicsData = readUS10kDemographics()
    df = saveFacialFeatures(demographicsData)
    # df = loadUS10KFacialFeatures()

    trainGP(df, scriptFolder, trainPercentage=0.8)