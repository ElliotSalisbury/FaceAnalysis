import numpy as np
import dlib
import cv2
import os

#initialize dlib detector
scriptFolder = os.path.realpath(__file__)
FACESWAP_SHAPEPREDICTOR_PATH = os.path.join(scriptFolder,"shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(FACESWAP_SHAPEPREDICTOR_PATH)

faceLines = np.load(os.path.join(scriptFolder,"lines.npy"))

def getNormalizingFactor(landmarks):
    hull = cv2.convexHull(landmarks)
    return np.sqrt(cv2.contourArea(hull))

def featuresFromLandmarks(landmarks):
    normalizingTerm = getNormalizingFactor(landmarks)
    normLandmarks = landmarks / normalizingTerm

    faceFeatures = normLandmarks[faceLines[:, 0]] - normLandmarks[faceLines[:, 1]]
    faceFeatures = np.linalg.norm(faceFeatures, axis=1)
    return faceFeatures

def getFaceFeatures(im):
    rects = detector(im, 1)
    if len(rects) != 1:
        return None, None

    landmarks = np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
    landmarks = np.array([[p[0, 0], p[0, 1]] for p in landmarks])

    faceFeatures = featuresFromLandmarks(landmarks)

    return landmarks, faceFeatures