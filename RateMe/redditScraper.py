import praw
import re
from imgurpython import ImgurClient
import urllib
import os
import sys
import csv
import time

def combineRatingCsvs(RateMeFolder):
    combinedPath = os.path.join(RateMeFolder, "combined.csv")

    submissionFolders = [x[0] for x in os.walk(RateMeFolder)]

    #compile into single file
    with open(combinedPath, 'w', newline='') as wf:
        writer = csv.writer(wf)
        writer.writerow(("Folder", "Submission Id", "Submission Created UTC", "Submission Title", "Submission Age", "Submission Gender", "Submission Author", "Rating Author", "Rating", "Decimal", "Rating Posted UTC", "Rating Text"))

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

def scrapeRateMe(dstFolder):
    my_user_agent = 'RateMeScraper'
    reddit_client_id = sys.argv[1]
    reddit_client_secret = sys.argv[2]

    imgur_client_id= sys.argv[3]
    imgur_client_secret = sys.argv[4]

    reddit = praw.Reddit(user_agent=my_user_agent,
                         client_id=reddit_client_id,
                         client_secret=reddit_client_secret)

    imgur = ImgurClient(imgur_client_id, imgur_client_secret)

    def getImgUrlsFromAlbum(albumId, is_gallery=False):
        imgurls = []
        try:
            if is_gallery:
                album = imgur.gallery_item(albumId)
            else:
                album = imgur.get_album(albumId)

            for image in album.images:
                imgurls.append(image['link'])
        except Exception as e:
            print(str(e))

        return imgurls

    def downloadImages(dstPath, imgurls):
        for imageurl in imgurls:
            try:
                filename = imageurl.split('/')[-1]
                filePath = os.path.join(dstPath, filename)
                urllib.request.urlretrieve(imageurl, filePath)
            except Exception as e:
                print(str(e))


    reAgeGender = re.compile("(\d+)[\s]*([MF])")
    reGenderAge = re.compile("([MF])[\s]*(\d+)")
    reRatings = [re.compile("(\d+)(\.\d+)? ?\/ ?10"),
                 re.compile("(\d+)(\.\d+)"),
                 re.compile("[sS]olid (\d+)(\.\d+)?")
                 ]
    reImgurAlbum = re.compile("imgur\.com\/a\/(\w+)")
    reImgurGallery = re.compile("imgur\.com\/gallery\/(\w+)")

    rateme = reddit.subreddit('rateme')
    submissions = rateme.submissions()
    # submissions = rateme.hot(limit=40)webs
    for submission in submissions:
        title = submission.title.upper()

        #get the gender and age
        result = reAgeGender.search(title)
        if result:
            age = result.group(1)
            gender = result.group(2)
        else:
            result = reGenderAge.search(title)
            if result:
                age = result.group(2)
                gender = result.group(1)
            else:
                continue

        submissionCreated = submission.created_utc
        submissionId = submission.id
        submissionTitle = submission.title.encode("ASCII", "ignore")
        submissionUps = submission.ups
        submissionDowns = submission.downs

        #skip the ones weve done already
        alreadyDone = False
        if os.path.exists(os.path.join(dstFolder,"%s_%s_%s"%(age,gender,submissionId))):
            alreadyDone = True

        #check the comments for ratings
        ratings = []
        noratings = []
        for top_level_comment in submission.comments:
            #ignore self comments
            if top_level_comment.author == "AutoModerator" or top_level_comment.author == submission.author:
                continue

            commentString = top_level_comment.body.encode("ASCII", "ignore")
            commentPosted = top_level_comment.created_utc
            commentUps = top_level_comment.ups
            commentDowns = top_level_comment.downs

            #try the different
            result = None
            for regex in reRatings:
                result = regex.search(top_level_comment.body)
                if result:
                    break

            if result:
                rating = result.group(1)
                decimal = result.group(2)
                if int(rating) <= 10:
                    ratingObject = (submissionId, submissionCreated, submissionTitle, age, gender, str(submission.author), str(top_level_comment.author), rating, decimal, commentPosted, commentString, submissionUps, submissionDowns, commentUps, commentDowns)
                    ratings.append(ratingObject)
                    continue


            noratingObject = (submissionId, submissionCreated, submissionTitle, age, gender, str(submission.author), str(top_level_comment.author), "", "", commentPosted, commentString, submissionUps, submissionDowns, commentUps, commentDowns)
            noratings.append(noratingObject)

        #if we have enough ratings then lets grab the images
        images = []
        if len(ratings) >= 3:
            dstPath = os.path.join(dstFolder,"%s_%s_%s" % (age, gender, submission.id))

            #only download images if we've not seen this submission before
            if not alreadyDone:
                imgurls = None
                if "imgur.com/a/" in submission.url:
                    url = submission.url
                    albumId = url.split("/")[-1]
                    imgurls = getImgUrlsFromAlbum(albumId)
                elif "imgur.com/gallery/" in submission.url:
                    url = submission.url
                    albumId = url.split("/")[-1]
                    imgurls = getImgUrlsFromAlbum(albumId, is_gallery=True)
                elif submission.is_self:
                    result = reImgurAlbum.search(submission.selftext)
                    if result:
                        imgurls = getImgUrlsFromAlbum(result.group(1))
                    else:
                        result = reImgurGallery.search(submission.selftext)
                        if result:
                            imgurls = getImgUrlsFromAlbum(result.group(1), is_gallery=True)
                else:
                    print(submission.url)
                    continue

                if imgurls:
                    if not os.path.exists(dstPath):
                        os.makedirs(dstPath)

                    downloadImages(dstPath, imgurls)
                    time.sleep(8)  # less than 500 calls per hour

            if alreadyDone or imgurls:
                #save the ratings file
                ratingsPath = os.path.join(dstPath,"ratings.csv")
                with open(ratingsPath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(("Submission Id", "Submission Created UTC", "Submission Title", "Submission Age", "Submission Gender", "Submission Author", "Rating Author", "Rating", "Decimal", "Rating Posted UTC", "Rating Text", "Submission Ups", "Submission Downs", "Rating Ups", "Rating Downs"))
                    writer.writerows(ratings)
                    writer.writerows(noratings)

def reCalcRatings(rateMeFolder):
    my_user_agent = 'RateMeScraper'
    reddit_client_id = sys.argv[1]
    reddit_client_secret = sys.argv[2]

    imgur_client_id= sys.argv[3]
    imgur_client_secret = sys.argv[4]

    reddit = praw.Reddit(user_agent=my_user_agent,
                         client_id=reddit_client_id,
                         client_secret=reddit_client_secret)


    reAgeGender = re.compile("(\d+)[\s]*([MF])")
    reGenderAge = re.compile("([MF])[\s]*(\d+)")
    reRatings = [re.compile("(\d+)(\.\d+)? ?\/ ?10"),
                 re.compile("(\d+)(\.\d+)"),
                 re.compile("[sS]olid (\d+)(\.\d+)?"),
                 re.compile("^(\d+)(\.\d+)?$"),
                 ]
    reImgurAlbum = re.compile("imgur\.com\/a\/(\w+)")
    reImgurGallery = re.compile("imgur\.com\/gallery\/(\w+)")

    submissionFolders = [x[0] for x in os.walk(rateMeFolder)][1:]

    # submissions = rateme.hot(limit=40)
    for folder in submissionFolders:
        submissionId = os.path.basename(folder).split("_")[-1]
        submission = reddit.submission(id=submissionId)

        title = submission.title.upper()

        # get the gender and age
        result = reAgeGender.search(title)
        if result:
            age = result.group(1)
            gender = result.group(2)
        else:
            result = reGenderAge.search(title)
            if result:
                age = result.group(2)
                gender = result.group(1)

        submissionCreated = submission.created_utc
        submissionId = submission.id
        submissionTitle = submission.title.encode("ASCII", "ignore")
        submissionUps = submission.ups
        submissionDowns = submission.downs

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(submissionCreated))
        print("{} {}".format(submissionId, timestamp))


        #check the comments for ratings
        ratings = []
        noratings = []
        for top_level_comment in submission.comments:
            #ignore self comments
            if top_level_comment.author == "AutoModerator" or top_level_comment.author == submission.author:
                continue

            commentString = top_level_comment.body.encode("ASCII", "ignore")
            commentPosted = top_level_comment.created_utc
            commentUps = top_level_comment.ups
            commentDowns = top_level_comment.downs

            #try the different
            result = None
            for regex in reRatings:
                result = regex.search(top_level_comment.body)
                if result:
                    break

            if result:
                rating = result.group(1)
                decimal = result.group(2)
                if int(rating) <= 10:
                    ratingObject = (submissionId, submissionCreated, submissionTitle, age, gender, str(submission.author), str(top_level_comment.author), rating, decimal, commentPosted, commentString, submissionUps, submissionDowns, commentUps, commentDowns)
                    ratings.append(ratingObject)
                    continue


            noratingObject = (submissionId, submissionCreated, submissionTitle, age, gender, str(submission.author), str(top_level_comment.author), "", "", commentPosted, commentString, submissionUps, submissionDowns, commentUps, commentDowns)
            noratings.append(noratingObject)

        #save the ratings file
        ratingsPath = os.path.join(folder,"ratings.csv")
        with open(ratingsPath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(("Submission Id", "Submission Created UTC", "Submission Title", "Submission Age", "Submission Gender", "Submission Author", "Rating Author", "Rating", "Decimal", "Rating Posted UTC", "Rating Text", "Submission Ups", "Submission Downs", "Rating Ups", "Rating Downs"))
            writer.writerows(ratings)
            writer.writerows(noratings)



if __name__ == "__main__":
    RateMeFolder = "E:\\Facedata\\RateMe"

    # reCalcRatings(RateMeFolder)
    scrapeRateMe(RateMeFolder)

    print("combining")
    combineRatingCsvs(RateMeFolder)