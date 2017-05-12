import glob
import os
import cv2
import pandas as pd
import pickle
from Beautifier.gaussianProcess import trainGP
from Beautifier.faceFeatures import getFaceFeatures
from Beautifier.face3D.faceFeatures3D import getMeshFromMultiLandmarks
from Beautifier.faceCNN.faceFeaturesCNN import getFaceFeaturesCNN
import numpy as np
import math

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

def reject_outliers(data, m = 2.):
    return data[abs(data - np.median(data)) <= m * np.std(data)]

scriptFolder = os.path.dirname(os.path.realpath(__file__))
def saveFacialFeatures(combinedcsvpath):
    df = pd.read_csv(combinedcsvpath)
    df.drop('Rating Text', 1)
    df.drop('Submission Title', 1)

    #filter out the weird ages
    filtered = df.loc[df['Submission Age'] >= 18]
    filtered = filtered.loc[filtered['Submission Age'] < 50]

    grouped = filtered.groupby(['Submission Id', "Submission Created UTC", "Folder", "Submission Age", "Submission Gender"])

    sfile = open(os.path.join(scriptFolder,"RateMeData_FULL.p"), "wb")
    pickler = pickle.Pickler(sfile)
    allData = []
    for i, (uniques, group) in enumerate(grouped):
        submissionId = uniques[0]
        submissionCreatedUTC = uniques[1]
        folder = uniques[2]
        submissionAge = uniques[3]
        submissionGender = uniques[4]

        ratings = np.array(group["Rating"].as_matrix().tolist())
        ratingDecimal = np.array(group["Decimal"].as_matrix().tolist())
        ratingAuthor = np.array(group["Rating Author"].as_matrix().tolist())
        ratingPostedUTC = np.array(group["Rating Posted UTC"].as_matrix().tolist())
        ratingText = np.array(group["Rating Text"].as_matrix().tolist())

        comments = [(ratingAuthor[k], ratingPostedUTC[k], ratings[k], ratingDecimal[k], ratingText[k]) for k in range(len(ratings))]

        cleanedRatings = [ratings[k] for k in range(len(ratings)) if not math.isnan(ratings[k])]
        cleanedRatings = reject_outliers(np.array(cleanedRatings))
        attractiveness = np.mean(cleanedRatings)

        #get the image files:
        types = ('*.jpg', '*.png', '*.bmp')
        impaths = []
        for type in types:
            impaths.extend(glob.glob(os.path.join(folder,type)))

        usedImPaths = []
        ims = []
        landmarkss = []
        faceFeaturess = []

        numBodyShots = 0
        numBuddyShots = 0
        for impath in impaths:
            im = cv2.imread(impath)
            # im = ensureImageLessThanMax(im)

            try:
                landmarks, faceFeatures = getFaceFeatures(im)

                usedImPaths.append(impath)
                ims.append(im)
                landmarkss.append(landmarks)
                faceFeaturess.append(faceFeatures)
            except Exception as e:
                if "No face" in str(e):
                    numBodyShots += 1
                elif "Multiple faces" in str(e):
                    numBuddyShots += 1

                continue

        if len(ims) > 0:
            print("(%i/%i) #i:%i" % (i, len(grouped), len(ims)))
            print(usedImPaths)
            meshs, poses, shape_coeffs, blendshape_coeffss = getMeshFromMultiLandmarks(landmarkss, ims, num_shape_coefficients_to_fit=-1, num_iterations=300)

            facefeaturesCNN = getFaceFeaturesCNN(ims, landmarkss)

            dataDict = {"submissionId":submissionId, "submissionCreatedUTC":submissionCreatedUTC, "gender": submissionGender, "age": submissionAge,
                        "comments": comments,
                        "attractiveness": attractiveness,
                        "numUsableImages": len(ims), "numSubmittedImages": len(impaths), "numBodyShots":numBodyShots, "numBuddyShots":numBuddyShots,
                        "impaths": usedImPaths,
                        "landmarkss": landmarkss, "facefeaturess": faceFeaturess,
                        # "meshs": meshs, can be regenerated from the coeffs below
                        "poses": poses, "facefeatures3D": shape_coeffs, "blendshape_coeffss": blendshape_coeffss,
                        "facefeaturesCNN": facefeaturesCNN,
                        }
            allData.append(dataDict)
            pickler.dump(dataDict)


    sfile.close()

    allDataDF = pd.DataFrame(allData)
    allDataDF = allDataDF.sample(frac=1).reset_index(drop=True)
    allDataDF.to_pickle(os.path.join(scriptFolder,"RateMeData.p"))

    return allDataDF

def dataFrameTo2D(df):
    import eos
    from Beautifier.face3D.faceFeatures3D import model, blendshapes, getFaceFeatures3D2DFromMesh
    allData = []
    for index, row in df.iterrows():
        gender = row["gender"]
        attractiveness = row["attractiveness"]
        facefeatures3D = row["facefeatures3D"]

        impaths = row["impaths"]
        landmarkss = row["landmarkss"]
        facefeaturess = row["facefeaturess"]
        poses = row["poses"]
        blendshape_coeffss = row["blendshape_coeffss"]

        for i in range(row["numUsableImages"]):
            impath = impaths[i]
            landmarks = landmarkss[i]
            facefeatures = facefeaturess[i]
            pose = poses[i]
            blendshape_coeffs = blendshape_coeffss[i]

            mesh = eos.morphablemodel.draw_sample(model, blendshapes, facefeatures3D, blendshape_coeffs, [])
            landmarks3D, facefeatures3D2D = getFaceFeatures3D2DFromMesh(mesh)

            dataDict = {"gender": gender, "attractiveness": attractiveness,
                        "landmarks": landmarks, "facefeatures": facefeatures, "landmarks3D":landmarks3D, "facefeatures3D2D": facefeatures3D2D,
                        "facefeatures3D": facefeatures3D, "pose":pose, "blendshape_coeffs": blendshape_coeffs,
                        "impath": impath}
            allData.append(dataDict)
    allDataDF = pd.DataFrame(allData)
    allDataDF = allDataDF.sample(frac=1).reset_index(drop=True)
    return allDataDF

def load(filename):
    with open(filename, "rb") as f:
        unpickler = pickle._Unpickler(f)
        while True:
            try:
                yield unpickler.load()
            except EOFError:
                break

def loadRateMeFacialFeatures():
    # stuff = load(os.path.join(scriptFolder, "RateMeData_FULL.p"))
    # df = pd.DataFrame(stuff)
    # df = df.sample(frac=1).reset_index(drop=True)
    df = pd.read_pickle(os.path.join(scriptFolder, "RateMeData.p"))
    return df
    # return pd.read_pickle(os.path.join(scriptFolder, "RateMeData.p"))
def loadRateMePCAGP(type="2d", gender="F"):
    return pickle.load(open(os.path.join(scriptFolder, "%s/GP_%s.p"%(type,gender)), "rb"))
def loadRateMe(type="2d", gender="F", server=False):
    if server:
        return pickle.load(open(os.path.join(scriptFolder, "server/%s_%s.p"%(type,gender)), "rb"))

    df = loadRateMeFacialFeatures()
    if type=="2d" or type=="3d2d":
        df = dataFrameTo2D(df)

    df_G = df.loc[df['gender'] == gender]

    featuresIndex = "facefeatures"
    if type=="3d":
        featuresIndex = "facefeatures3D"
    elif type=="3d2d":
        featuresIndex = "facefeatures3D2D"
    elif type == "cnn":
        featuresIndex = "facefeaturesCNN"

    trainX = np.array(df_G[featuresIndex].as_matrix().tolist())
    trainY = np.array(df_G["attractiveness"].as_matrix().tolist())
    pca, gp = loadRateMePCAGP(type=type, gender=gender)

    return trainX, trainY, pca, gp

def saveServerOptimised():
    for type in ["2d", "3d"]:
        for gender in ["F", "M"]:
            with open(os.path.join(scriptFolder, "server/%s_%s.p"%(type,gender)), "wb") as file:
                pickle.dump(loadRateMe(type=type, gender=gender), file)

if __name__ == "__main__":
    rateMeFolder = "E:\\Facedata\\RateMe"
    combinedPath = os.path.join(rateMeFolder, "combined.csv")

    df = saveFacialFeatures(combinedPath)
    # df = loadRateMeFacialFeatures()


    # df2d = dataFrameTo2D(df)
    # trainGP(df2d, os.path.join(scriptFolder, "2d"), trainPercentage=0.9, train_on_PCA=False, generate_PCA=True)
    # trainGP(df2d, os.path.join(scriptFolder, "3d2d"), trainPercentage=0.9, featureset="facefeatures3D2D", train_on_PCA=False, generate_PCA=True)
    #
    # moreaccurate = df[df["numUsableImages"]>=1]
    # trainGP(moreaccurate, os.path.join(scriptFolder, "3d"), trainPercentage=0.9, featureset="facefeatures3D", train_on_PCA=False, generate_PCA=False)
    #
    # trainGP(df, os.path.join(scriptFolder, "cnn"), trainPercentage=0.9, featureset="facefeaturesCNN", train_on_PCA=False, generate_PCA=False)
    #
    # saveServerOptimised()