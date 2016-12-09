import praw
import re
from imgurpython import ImgurClient
import urllib
import os
import sys
import csv

my_user_agent = 'RateMeScraper'
reddit_client_id = sys.argv[1]
reddit_client_secret = sys.argv[2]

imgur_client_id= sys.argv[3]
imgur_client_secret = sys.argv[4]

reddit = praw.Reddit(user_agent=my_user_agent,
                     client_id=reddit_client_id,
                     client_secret=reddit_client_secret)

imgur = ImgurClient(imgur_client_id, imgur_client_secret)

def getImgUrlsFromAlbum(albumId):
    album = imgur.get_album(albumId)

    imgurls = []
    for image in album.images:
        imgurls.append(image['link'])
    return imgurls

def downloadImages(dstPath, imgurls):
    for imageurl in imgurls:
        filename = imageurl.split('/')[-1]
        filePath = os.path.join(dstPath, filename)
        urllib.urlretrieve(imageurl, filePath)


reAgeGender = re.compile("(\d+)[\s]*([M|F])")
reGenderAge = re.compile("([M|F])[\s]*(\d+)")
reRatingSlash = re.compile("(\d+)(\.\d+)?\/10")
reImgurAlbum = re.compile("imgur\.com\/a\/(\w+)")

rateme = reddit.subreddit('rateme')
# submissions = rateme.submissions()
submissions = rateme.hot(limit=40)
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

    #check the comments for ratings
    ratings = []
    for top_level_comment in submission.comments:
        result = reRatingSlash.search(top_level_comment.body)
        if result:
            rating = result.group(1)
            decimal = result.group(2)
            if int(rating) <= 10:
                ratingObject = (age,gender,submission.author.name,top_level_comment.author.name, rating, decimal)
                ratings.append(ratingObject)

    #if we have enough ratings then lets grab the images
    images = []
    if len(ratings) >= 3:
        imgurls = None
        if "imgur.com/a/" in submission.url:
            url = submission.url
            albumId = url.split("/")[-1]
            imgurls = getImgUrlsFromAlbum(albumId)
        elif submission.is_self:
            result = reImgurAlbum.search(submission.selftext)
            if result:
                imgurls = getImgUrlsFromAlbum(result.group(1))
        else:
            print(submission.url)
            continue

        if imgurls:
            dstPath = "%s_%s_%s"%(age,gender,submission.id)
            if not os.path.exists(dstPath):
                os.makedirs(dstPath)

            downloadImages(dstPath, imgurls)

            #save the ratings file
            ratingsPath = os.path.join(dstPath,"ratings.csv")
            with open(ratingsPath, 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(("Age","Gender","RatedUsername","RatingUsername","Rating","Decimal"))
                writer.writerows(ratings)