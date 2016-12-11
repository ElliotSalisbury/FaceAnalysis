import os
import csv
import pandas as pd
import cv2
import numpy as np
from calculateFaceData import ensureImageLessThanMax
from faceFeatures import getFaceFeatures
from beautifier import findBestFeaturesKNN,calculateLandmarksfromFeatures
from warpFace import warpFace

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

def averageFaces(df):
    im = cv2.imread(impath)
    im = ensureImageLessThanMax(im, maxsize=256)

    hotnessRange = range(4,10)

    landmarks, faceFeatures = getFaceFeatures(im)

    imfaces = np.zeros((im.shape[0]*2, im.shape[1]*len(hotnessRange), im.shape[2]), dtype=np.uint8)

    grouped = df.groupby("gender")
    for i, (gender, group) in enumerate(grouped):
        for j, hotness in enumerate(hotnessRange):
            hotgroup = group.loc[group['attractiveness'] >= hotness]
            hotgroup = hotgroup.loc[hotgroup['attractiveness'] < hotness+0.5]

            hotFacefeatures = np.array(hotgroup["facefeatures"].as_matrix().tolist())

            print("%s %d %d"%(gender,hotness,hotFacefeatures.shape[0]))

            hotFacefeatures = np.mean(hotFacefeatures, axis=0)

            newLandmarksKNN = calculateLandmarksfromFeatures(landmarks, hotFacefeatures)
            hotFace = warpFace(im, landmarks, newLandmarksKNN)

            imfaces[im.shape[0]*i:im.shape[0]*(i+1), im.shape[1]*j:im.shape[1]*(j+1)] = hotFace
    cv2.imshow("sdfs", imfaces)
    cv2.waitKey(-1)



if __name__ == "__main__":
    #load in the dataframes for analysis
    df = pd.read_pickle("RateMeData.p")

    averageFaces(df)