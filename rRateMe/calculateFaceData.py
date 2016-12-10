import glob
import os
from gaussianProcess import trainGP
import cv2
import pandas as pd

from Beautifier.faceFeatures import getFaceFeatures

MAX_IM_SIZE = 512

def ensureImageLessThanMax(im):
    height, width, depth = im.shape
    if height > MAX_IM_SIZE or width > MAX_IM_SIZE:

        if width > height:
            ratio = MAX_IM_SIZE / float(width)
            width = MAX_IM_SIZE
            height = int(height * ratio)
        else:
            ratio = MAX_IM_SIZE / float(height)
            height = MAX_IM_SIZE
            width = int(width * ratio)
        im = cv2.resize(im,(width,height))
    return im

saveFolder = "C:\\Users\\ellio\\PycharmProjects\\circlelines\\rRateMe"
def saveFacialFeatures(combinedcsvpath):
    df = pd.read_csv(combinedcsvpath)
    df.drop('Rating Text', 1)
    df.drop('Submission Title', 1)

    #filter out the weird ages
    filtered = df.loc[df['Submission Age'] >= 18]

    grouped = filtered.groupby(['Submission Gender', "Folder"])

    allData = []
    for genderfolder, group in grouped:
        gender = genderfolder[0]
        folder = genderfolder[1]
        rating = group["Rating"].mean()

        #get the image files:
        types = ('*.jpg', '*.png', '*.bmp')
        impaths = []
        for type in types:
            impaths.extend(glob.glob(os.path.join(folder,type)))

        for impath in impaths:
            im = cv2.imread(impath)
            im = ensureImageLessThanMax(im)

            landmarks, faceFeatures = getFaceFeatures(im)

            if faceFeatures is not None:
                dataDict = {"gender":gender,"attractiveness":rating,"landmarks":landmarks,"facefeatures":faceFeatures,"impath":impath}
                allData.append(dataDict)

    allDataDF = pd.DataFrame(allData)
    allDataDF = allDataDF.sample(frac=1).reset_index(drop=True)
    allDataDF.to_pickle(os.path.join(saveFolder,"RateMeData.p"))

    return allDataDF

def loadRateMeFacialFeatures():
    return pd.read_pickle(os.path.join(saveFolder, "RateMeData.p"))

if __name__ == "__main__":
    rateMeFolder = "E:\\Facedata\\RateMe"
    combinedPath = os.path.join(rateMeFolder, "combined.csv")

    df = saveFacialFeatures(combinedPath)
    # df = loadRateMeFacialFeatures()

    trainGP(df, saveFolder)