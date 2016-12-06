import csv
import json
from sklearn import gaussian_process
import numpy as np
import cv2

data = []
with open('features','r') as featuref:
    reader = csv.reader(featuref)
    for row in reader:
        impath = row[0]
        features = json.loads(row[1])
        score = float(row[2])

        data.append([impath,features,score])
trainSize = int(len(data)*0.8)

X = np.zeros((len(data),4096),dtype=np.float32)
Y = np.zeros((len(data),1),dtype=np.float32)
for i, datum in enumerate(data):
    X[i, :] = datum[1]
    Y[i, 0] = datum[2]

trainX = X[:trainSize]
trainY = Y[:trainSize]
testX = X[trainSize:]
testY = Y[trainSize:]

gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
gp.fit(trainX, trainY)

y_pred, sigma2_pred = gp.predict(testX, eval_MSE=True)

error = np.zeros((len(testY),2),dtype=np.float32)
error[:,0] = testY[:,0]
error[:,1] = y_pred[:,0]

sortedIndexs = np.argsort(y_pred,0)
for i, index in enumerate(sortedIndexs):
    impath = data[trainSize+index][0]
    crowdscore = data[trainSize+index][2]
    predictedscore = y_pred[index]

    im = cv2.imread(impath)
    cv2.imwrite("./imgs/%d_%0.2f_%0.2f_%0.2f.jpg"%(i,crowdscore,predictedscore,crowdscore-predictedscore), im)

notscoreddata = []
with open('featuresnotscored', 'r') as featuref:
    reader = csv.reader(featuref)
    for row in reader:
        impath = row[0]
        features = json.loads(row[1])
        score = float(row[2])

        notscoreddata.append([impath, features, score])
notScoredX = np.zeros((len(data),4096),dtype=np.float32)
for i, datum in enumerate(data):
    notScoredX[i, :] = datum[1]

y_pred, sigma2_pred = gp.predict(notScoredX, eval_MSE=True)

sortedIndexs = np.argsort(y_pred,0)
for i, index in enumerate(sortedIndexs):
    impath = notscoreddata[trainSize+index][0]
    crowdscore = notscoreddata[trainSize+index][2]
    predictedscore = y_pred[index]

    im = cv2.imread(impath)
    cv2.imwrite("./imgsnotscored/%d_%0.2f_%0.2f_%0.2f.jpg"%(i,crowdscore,predictedscore,crowdscore-predictedscore), im)
