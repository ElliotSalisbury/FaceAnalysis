import numpy as np
import dlib
import cv2
import os

#initialize dlib detector
scriptFolder = os.path.dirname(os.path.realpath(__file__))
DLIB_SHAPEPREDICTOR_PATH = os.environ['DLIB_SHAPEPREDICTOR_PATH']

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_SHAPEPREDICTOR_PATH)

faceLines = np.load(os.path.join(scriptFolder,"lines.npy"))

def getLandmarks(im):
    rects = detector(im, 1)
    if len(rects) != 1:
        return None

    landmarks = np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
    landmarks = np.array([[p[0, 0], p[0, 1]] for p in landmarks])

    return landmarks

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
    landmarks = getLandmarks(im)

    if landmarks is None:
        return None, None

    faceFeatures = featuresFromLandmarks(landmarks)

    return landmarks, faceFeatures