import praw
import re
from imgurpython import ImgurClient
import urllib
import os
import sys
import csv
import time

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
            urllib.urlretrieve(imageurl, filePath)
        except Exception as e:
            print(str(e))


reAgeGender = re.compile("(\d+)[\s]*([M|F])")
reGenderAge = re.compile("([M|F])[\s]*(\d+)")
reRatingSlash = re.compile("(\d+)(\.\d+)?\/10")
reImgurAlbum = re.compile("imgur\.com\/a\/(\w+)")
reImgurGallery = re.compile("imgur\.com\/gallery\/(\w+)")

rateme = reddit.subreddit('rateme')
submissions = rateme.submissions()
# submissions = rateme.hot(limit=40)
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

    #skip the ones weve done already
    if os.path.exists("./data/%s_%s_%s"%(age,gender,submission.id)):
        continue

    #check the comments for ratings
    ratings = []
    for top_level_comment in submission.comments:
        result = reRatingSlash.search(top_level_comment.body)
        if result:
            rating = result.group(1)
            decimal = result.group(2)
            if int(rating) <= 10:
                titleString = submission.title.encode("ASCII", "ignore")
                commentString = top_level_comment.body.encode("ASCII", "ignore")

                ratingObject = (titleString, age, gender, str(submission.author), str(top_level_comment.author), rating, decimal, commentString)
                ratings.append(ratingObject)

    #if we have enough ratings then lets grab the images
    images = []
    if len(ratings) >= 3:
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
            dstPath = "./data/%s_%s_%s"%(age,gender,submission.id)
            if not os.path.exists(dstPath):
                os.makedirs(dstPath)

            downloadImages(dstPath, imgurls)

            #save the ratings file
            ratingsPath = os.path.join(dstPath,"ratings.csv")
            with open(ratingsPath, 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(("Submission Title", "Submission Age","Submission Gender","Submission Author","Rating Author","Rating","Decimal", "Rating Text"))
                writer.writerows(ratings)

        time.sleep(8) # less than 500 calls per hour