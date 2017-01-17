import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import collections
import cv2
from Beautifier.faceFeatures import getLandmarks
from Beautifier.warpFace import drawLandmarks
from Beautifier.face3D.faceFeatures3D import getMeshFromLandmarks
from Beautifier.face3D.warpFace3D import drawMesh

# titleFont = fm.FontProperties(fname="./fonts/Oswald-Regular.ttf")
# titleFont = fm.FontProperties(fname="./fonts/Oswald-Light.ttf")
titleFont = fm.FontProperties(fname="./fonts/Pacifico-Regular.ttf")

genderName = {"M":"Male","F":"Female"}
genderColor = {"M":"#2196f3","F":"#862197"}


def histOfAttractiveness(df):
    df = df[np.isfinite(df['Rating'])]

    gendered = df.groupby(['Submission Gender'])
    for gender, group in gendered:

        title = genderName[gender] + " Ratings"

        ax = group["Rating"].hist(bins=np.arange(1,12)-0.5, color=genderColor[gender], grid=False)
        ax.set_title(title, fontproperties=titleFont)
        ax.set_xticks(range(1,11))
        ax.set_xlim(0.5,10.5)

        plt.savefig(title)
        plt.clf()

    gendered = df.groupby(['Submission Gender'])
    for gender, group in gendered:
        ratingsPerSubmission = {}
        perSubmission = group.groupby(['Folder'])
        for submission, group in perSubmission:
            submissionMean = []
            for index, row in group.iterrows():
                submissionMean.append(int(row["Rating"]))
            ratingsPerSubmission[submission] = submissionMean
        ratings = [np.mean(reject_outliers(np.array(ratingsPerSubmission[submission]))) for submission in ratingsPerSubmission.keys()]
        plt.hist(ratings, bins=np.arange(1, 12) - 0.5, color=genderColor[gender])

        title = genderName[gender] + " Attractiveness"
        plt.title(title, fontproperties=titleFont)
        plt.xticks(range(1, 11))
        plt.xlim(0.5, 10.5)

        plt.savefig(title)
        plt.clf()

        ratingsNoOutliers = [reject_outliers(np.array(ratingsPerSubmission[submission])) for submission in ratingsPerSubmission.keys()]
        ratingsNoOutliers = [item for sublist in ratingsNoOutliers for item in sublist]
        plt.hist(ratingsNoOutliers, bins=np.arange(1, 12) - 0.5, color=genderColor[gender])
        title = genderName[gender] + " Ratings without outliers"
        plt.title(title, fontproperties=titleFont)
        plt.xticks(range(1, 11))
        plt.xlim(0.5, 10.5)
        plt.savefig(title)
        plt.clf()


        width = 1
        hw = width / 2.0
        numRatings = [len(ratingsPerSubmission[submission]) for submission in ratingsPerSubmission.keys()]
        plt.hist(numRatings, bins=np.arange(1, 50,width) - hw, color=genderColor[gender])

        title = genderName[gender] + " Submissions Number of Ratings"
        plt.title(title, fontproperties=titleFont)
        plt.xticks(range(1, 51,width*2))
        plt.xlim(hw, 50 + hw)

        plt.savefig(title)
        plt.clf()



        print(genderName[gender] + " #Submissions: " + str(len(ratingsPerSubmission.keys())))

def topRatingAuthors(df):
    authorRatings = collections.defaultdict(list)
    for index, row in df.iterrows():
        authorRatings[row['Rating Author']].append(row["Rating"])

    authors = list(authorRatings.keys())
    authors = sorted(authors, key=lambda author: len(authorRatings[author]))

    top = authors[-10:]

    data = np.random.lognormal(size=(37, 4), mean=1.5, sigma=1.75)
    myData = []
    labels = []
    for author in top:
        if author == "None":
            continue
        myData.append(authorRatings[author])
        labels.append(author)

    plt.boxplot(myData, labels=labels)
    plt.savefig("Top Commenters Distribution of Ratings")
    plt.clf()

def ratingsOverAge(df):
    df = df[np.isfinite(df['Rating'])]
    df = df.loc[df['Submission Age'] <= 30]

    gendered = df.groupby(['Submission Gender'])
    for gender, group in gendered:
        ageRatings = collections.defaultdict(list)
        perSubmission = group.groupby(['Folder'])

        for submission, group in perSubmission:
            submissionMean = []
            for index, row in group.iterrows():
                submissionMean.append(int(row["Rating"]))
            ageRatings[int(row['Submission Age'])].append(np.mean(reject_outliers(np.array(submissionMean))))

        X = []
        Y = []
        std = []
        for age in ageRatings:
            X.append(age)
            Y.append(np.mean(ageRatings[age]))
            std.append(np.std(ageRatings[age]))

        title = genderName[gender] + " Attractiveness Over Age"
        plt.errorbar(X, Y, yerr=std, fmt='-', color=genderColor[gender])
        # plt.plot(X,Y)
        plt.title(title, fontproperties=titleFont)
        plt.ylim(0,10)
        plt.savefig(title)
        plt.clf()

def numRatersWhoSubmitted(df):
    df = df[np.isfinite(df['Rating'])]

    submittersCount = collections.defaultdict(set)
    ratersCount = collections.defaultdict(set)
    ratingsByAuthor = collections.defaultdict(list)
    for index, row in df.iterrows():
        submittersCount[row["Submission Author"]].add(row["Folder"])
        ratersCount[row["Rating Author"]].add(row["Folder"])
        ratingsByAuthor[row["Rating Author"]].append(int(row["Rating"]))

    submitters = set(submittersCount.keys())
    raters = set(ratersCount.keys())
    both = raters.intersection(submitters)
    noSubmitters = raters-submitters

    print("#Submitters :%d, #Raters :%d, #Both :%d"%(len(submitters), len(raters), len(both)))

    numRepeatSubmissions = []
    for submitter in submittersCount:
        numRepeatSubmissions.append(len(submittersCount[submitter]))

    #HIST of the submission count
    plt.hist(numRepeatSubmissions, bins=np.arange(1,np.max(numRepeatSubmissions)+1)-0.5)
    title = "Number of Submissions Per User"
    plt.title(title, fontproperties=titleFont)
    plt.xticks(range(1,np.max(numRepeatSubmissions)+1))
    # plt.ylim(0,200)
    plt.xlim(0.5, np.max(numRepeatSubmissions)+0.5)
    plt.savefig(title)
    plt.clf()

    #box and whiskers wether submitters rate differently to others (empathy?)
    bothRatings = []
    noSubmittersRatings = []
    for rater in both:
        bothRatings.extend(ratingsByAuthor[rater])
    for rater in noSubmitters:
        noSubmittersRatings.extend(ratingsByAuthor[rater])
    labels = ["Submitters", "No Submitters"]
    data = [bothRatings, noSubmittersRatings]
    plt.boxplot(data, labels=labels)
    plt.savefig("Submitters Vs NoSubmmiters Distribution of Ratings")
    plt.clf()

    #variance of raters
    variance = []
    for rater in raters:
        if len(ratingsByAuthor[rater]) < 3:
            continue
        variance.append(np.var(ratingsByAuthor[rater]))
    plt.hist(variance, bins=np.arange(0,4,0.1))
    plt.savefig("Variance of Raters")
    plt.clf()

def reject_outliers(data, m = 2.):
    return data[abs(data - np.median(data)) <= m * np.std(data)]


def rankSubmissions(df):
    df = df[np.isfinite(df['Rating'])]

    gendered = df.groupby(['Submission Gender'])
    for gender, group in gendered:
        ratingsPerSubmission = {}
        perSubmission = group.groupby(['Folder'])
        for submission, group in perSubmission:
            submissionMean = []
            for index, row in group.iterrows():
                submissionMean.append(int(row["Rating"]))
            data = reject_outliers(np.array(submissionMean))
            if data.any():
                ratingsPerSubmission[submission] = data

            # ratingsPerSubmission = {submission:reject_outliers(np.array(ratingsPerSubmission[submission])) for submission in ratingsPerSubmission}
        sortedSubmissions = sorted(ratingsPerSubmission.keys(), key=lambda submission: np.mean(ratingsPerSubmission[submission]))
        # sortedSubmissions = [submission for submission in sortedSubmissions if len(ratingsPerSubmission[submission]) > 5]
        # sortedSubmissions = [submission for submission in sortedSubmissions if np.mean(ratingsPerSubmission[submission]) >= 8]

        X = np.arange(len(sortedSubmissions))
        Ym = np.array([np.mean(ratingsPerSubmission[submission]) for submission in sortedSubmissions])
        Yerr = np.array([np.std(ratingsPerSubmission[submission]) for submission in sortedSubmissions])


        title = genderName[gender] + " Submissions Sorted by Average Rating"
        plt.plot(X, Ym, color=genderColor[gender])
        plt.plot(X, Ym + Yerr, 'r-')
        plt.plot(X, Ym - Yerr, 'r-')
        plt.title(title, fontproperties=titleFont)
        plt.ylim(0, 10)
        plt.savefig(title)
        plt.clf()

def imagesOfFitting():
    im = cv2.imread("C:\\Users\\ellio\\Desktop\\lena.bmp")
    landmarks = getLandmarks(im)

    cv2.imwrite("Landmarks.jpg", drawLandmarks(im, landmarks))

    mesh, pose, _, _ = getMeshFromLandmarks(landmarks, im)
    cv2.imwrite("Mesh.jpg", drawMesh(im, mesh, pose))

if __name__ == "__main__":
    rateMeFolder = "E:\\Facedata\\RateMe"
    combinedPath = os.path.join(rateMeFolder, "combined.csv")
    submissionsdf = pd.read_csv(combinedPath)
    submissionsdf.drop('Rating Text', 1)
    submissionsdf.drop('Submission Title', 1)

    # filter out the weird ages
    submissionsdf = submissionsdf.loc[submissionsdf['Submission Age'] >= 18]
    submissionsdf = submissionsdf.loc[submissionsdf['Submission Age'] < 50]

    df = submissionsdf

    histOfAttractiveness(df)
    topRatingAuthors(df)
    ratingsOverAge(df)
    numRatersWhoSubmitted(df)
    rankSubmissions(df)
    # imagesOfFitting()