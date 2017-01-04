import numpy as np
import cv2
import os
from faceFeatures import getFaceFeatures
import pickle
from beautifier import beautifyFace
from RateMe import loadRateMeFacialFeatures

scriptFolder = os.path.dirname(os.path.realpath(__file__))
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

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    pca, gp = pickle.load(
        open(os.path.join(scriptFolder, "../RateMe/GP_F.p"), "rb"))

    ratemedf = loadRateMeFacialFeatures()
    ratemegendered = ratemedf.loc[ratemedf['gender'] == 'F']

    #split into training sets
    trainX = np.array(ratemegendered["facefeatures"].as_matrix().tolist())
    trainY = np.array(ratemegendered["attractiveness"].as_matrix().tolist())

    NUM_BEST = 5
    bestimgs = []
    scores = []
    count = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        frame = ensureImageLessThanMax(frame)

        landmarks, features = getFaceFeatures(frame)

        if features is not None:
            # hotterFace = beautifyFace(frame, landmarks, features, pca, gp, trainX, trainY, method='KNN')

            # draw the landmarks


            reducedFeatures = pca.transform(features)

            # score, std = gp.predict(reducedFeatures, return_std=True)
            score, std = gp.predict(reducedFeatures, return_std=False),0
            scores.append([score, std])
            scores = scores[-20:]
            avg = np.mean(np.array(scores), axis=0)

            bestimgs.append([score,frame.copy()])
            bestimgs = sorted(bestimgs, key=lambda e:-e[0])[:NUM_BEST]

            for i, landmark in enumerate(landmarks):
                p = (int(landmark[0]), int(landmark[1]))
                cv2.circle(frame, p, 3, (0, 255, 255), thickness=-1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "  %0.2f +- %0.2f"%(avg[0], avg[1]), p, font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.imshow("hotter", hotterFace)

        #display best imgs
        bestImgsIm = np.zeros((frame.shape[0], frame.shape[1]*NUM_BEST, frame.shape[2]), dtype=np.uint8)
        for i, bestImg in enumerate(bestimgs):
            im = bestImg[1]
            bestImgsIm[:,i*frame.shape[1]:(i+1)*frame.shape[1],:] = im


        # Display the resulting frame
        cv2.imshow('frame', frame)
        cv2.imshow('best', bestImgsIm)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()