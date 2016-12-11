import numpy as np
import cv2
from faceFeatures import getFaceFeatures
import pickle
from beautifier import findBestFeaturesKNN,calculateLandmarksfromFeatures
from warpFace import warpFace
from US10K import loadUS10KFacialFeatures

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

    gp = pickle.load(
        open("C:\\Users\\Elliot\\PycharmProjects\\FaceAnalysis\\US10K\\GP_M.p", "rb"))
    us10kdf = loadUS10KFacialFeatures()

    us10kwomen = us10kdf.loc[us10kdf['gender'] == 'M']

    #split into training sets
    trainSize = int(us10kwomen.shape[0] * 0.8)
    traindf = us10kwomen[:trainSize]
    trainX = np.array(traindf["facefeatures"].as_matrix().tolist())
    trainY = np.array(traindf["attractiveness"].as_matrix().tolist())

    scores = []
    count = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        frame = ensureImageLessThanMax(frame)

        landmarks, features = getFaceFeatures(frame)


        if features is not None:
            if count %60 == 0:
                optimalNewFaceFeaturesKNN = findBestFeaturesKNN(features, gp, trainX, trainY)
            count+=1
            newLandmarksKNN = calculateLandmarksfromFeatures(landmarks, optimalNewFaceFeaturesKNN)
            beautifulFaceKNN = warpFace(frame, landmarks, newLandmarksKNN)


            # draw the landmarks
            for i, landmark in enumerate(landmarks):
                p = (int(landmark[0]), int(landmark[1]))
                cv2.circle(frame, p, 3, (0, 255, 255), thickness=-1)

            score = gp.predict([features])[0]
            scores.append(score)
            scores = scores[-20:]
            avg = sum(scores) / len(scores)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "%0.3f"%avg, p, font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('frame',beautifulFaceKNN)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()