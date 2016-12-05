import fnmatch
import os
import dlib
import cv2
import json
from bs4 import BeautifulSoup
import re
import csv
import collections
import numpy as np
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()
metafile = "C:\\Users\\ellio\\Desktop\\fb\\Elliot Salisbury_all_meta.txt"
srcDir = "C:\\Users\\ellio\\Desktop\\fb\\Elliot Salisbury_all_backup_files"
dstDir = "C:\\Users\\ellio\\Desktop\\fb\\MyFaces"
impaths = []

trainingcsv = os.path.join(dstDir, "trainingdata.csv")
trainingcsvf = open(trainingcsv,'w',newline='')
trainingcsvwriter = csv.writer(trainingcsvf)
# mturkimhost = "https://crowddrone.ecs.soton.ac.uk:9090/static/%s"

#process the mturk rating results
mturkresultspath = "C:\\Users\\ellio\\Desktop\\fb\\Batch_2618752_batch_results.csv"
mturkresults = collections.defaultdict(list)
INVALID = 'invalid'
with open(mturkresultspath, 'r') as mturkresultsf:
    mturkcsvreader = csv.reader(mturkresultsf)
    for i, row in enumerate(mturkcsvreader):
        if i == 0:
            continue

        if row[30] == INVALID:
            mturkresults[int(row[27])].append(INVALID)
        else:
            mturkresults[int(row[27])].append(int(row[31]))
#filter the images that have more than a third INVALIDS, and remove INVALIDS from the list
mturkresults = {key:[score for score in mturkresults[key] if score != INVALID] for key in mturkresults if mturkresults[key].count(INVALID) / float(len(mturkresults[key])) < 0.33}

# scores = []
# for imgid in mturkresults:
#     score = np.mean(mturkresults[imgid])
#     scores.append(score)
# n, bins, patches = plt.hist(scores, 20, normed=1, facecolor='green', alpha=0.75)
# plt.show()

#read meta data file
with open(metafile, 'r', encoding="utf8") as data_file:
    jsonline = data_file.readline()

    start = ",\"photos\":"
    end = ",\"type\":"
    start = jsonline.index(start) + len(start)
    end = jsonline.index(end, start)
    data = jsonline[start:end]
    metadata = json.loads(data)

#read all image paths
for root, dirnames, filenames in os.walk(srcDir):
    for filename in fnmatch.filter(filenames, '*.jpg'):
        impaths.append(os.path.join(root, filename))


for i, data in enumerate(metadata):
    #add the mturk results to the metadata
    if i in mturkresults:
        data["mturkrating"] = mturkresults[i]

    # find the image path for this metadata
    url = data["href"]
    #two different url schemes exist, try both
    try:
        imgid = url[url.index("fbid=")+5:]
    except ValueError as ve:
        #fbid not found, try different method
        betweenslashes = url.split("/")
        imgid = betweenslashes[-2]

    #check that we havent found another image in our dataset that matches this id
    matches = []
    for impath in impaths:
        if imgid in impath:
            matches.append(impath)
    if len(matches) != 1:
        raise Exception("no matches, or multiple matches found to image id")

    data["impath"] = matches[0]

    #parse the tags
    tagHTML = data["tag"]

    soup = BeautifulSoup(tagHTML, "html.parser")

    #check each tag looking for me
    for tagxml in soup.findAll("div", "fbPhotosPhotoTagboxBase"):
        try:
            tagname = tagxml.find("div", "tagName").contents[0].contents[0]
        except:
            continue
        if tagname == "Elliot Salisbury":
            #parse the style to get the width and height
            tagLoc = tagxml["style"]
            tagLoc = tagLoc.replace("%","")
            tagLocs = re.split(":|;", tagLoc)
            def parsePer(s):
                return float(s)/100.0
            tw,th,tx,ty = parsePer(tagLocs[1]),parsePer(tagLocs[3]),parsePer(tagLocs[5]),parsePer(tagLocs[7])

            # expand the region a bit
            expansion = 0.1
            ew = tw * expansion
            eh = th * expansion
            tx = max(0, tx - ew)
            ty = max(0, ty - eh)
            tw = tw + ew
            th = th + eh
            if tx + tw > 1:
                tw = 1.0 - tx
            if ty + th > 1:
                th = 1.0 - ty

            #read the image and get the location of tag in pixel coords
            im = cv2.imread(data["impath"])
            ih,iw,id = im.shape

            px1 = int(iw * tx)
            px2 = int(iw * tw) + px1
            py1 = int(ih * ty)
            py2 = int(ih * th) + py1

            faceim = im[py1:py2, px1:px2].copy()

            #detect faces in the face subimage
            rects = detector(faceim, 1)
            if len(rects) >= 1:
                # for rect in rects:
                #     ry = rect.top()
                #     rx = rect.left()
                #     rw = rect.width()
                #     rh = rect.height()
                #     cv2.rectangle(faceim,(rx,ry),(rx+rw,ry+rh),(255,0,0))
                #
                #     cv2.imshow("test", faceim)
                #     cv2.waitKey(1)

                filename ="%04d.jpg"%i
                # cv2.imwrite(os.path.join(dstDir,filename), faceim)
                # imhost = mturkimhost % filename
                # mturkcsvwriter.writerow([i, imgid, imhost])

                #get images ready for training
                if "mturkrating" in data and len(data["mturkrating"]) > 1:
                    median = np.median(data["mturkrating"])
                    mean = np.mean(data["mturkrating"])

                    outpath = os.path.join(dstDir, str(median))
                    if not os.path.exists(outpath):
                        os.makedirs(outpath)

                    facepath = os.path.join(outpath, filename)

                    #resize for neural net
                    faceim = cv2.resize(faceim, (227,227))
                    cv2.imwrite(facepath, faceim)

                    trainingcsvwriter.writerow([facepath, median, mean])

trainingcsvf.close()