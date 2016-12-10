import numpy as np
import dlib

#initialize dlib detector
FACESWAP_SHAPEPREDICTOR_PATH = "C:\\Users\\ellio\\PycharmProjects\\circlelines\\Beautifier\\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(FACESWAP_SHAPEPREDICTOR_PATH)

faceLines = np.load("C:\\Users\\ellio\\PycharmProjects\\circlelines\\Beautifier\\lines.npy")

def getFaceFeatures(im):
    rects = detector(im, 1)
    if len(rects) != 1:
        return None, None

    landmarks = np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
    landmarks = np.array([[p[0, 0], p[0, 1]] for p in landmarks])

    # normalizingTerm = (landmarks.max(axis=0) - landmarks.min(axis=0))
    normalizingTerm = np.linalg.norm(landmarks[0] - landmarks[16])  # facewidth
    normLandmarks = landmarks / normalizingTerm

    faceFeatures = normLandmarks[faceLines[:, 0]] - normLandmarks[faceLines[:, 1]]
    faceFeatures = np.linalg.norm(faceFeatures, axis=1)

    return landmarks, faceFeatures