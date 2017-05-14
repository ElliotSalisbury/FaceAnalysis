import csv
import collections
import json
import os
import cv2
import matplotlib.pyplot as plt
import glob
from Beautifier.faceFeatures import getLandmarks
from Beautifier.face3D.warpFace3D import warpFace3D
from Beautifier.face3D.faceFeatures3D import getMeshFromLandmarks, model, blendshapes
import numpy as np

def readMturkResults(filepath):
    results_person_coeffs = {}

    with open(filepath, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)

        for i, row in enumerate(reader):
            if i == 0:
                continue

            workerId = row[14]
            imagelist = row[27]
            leastIndex = row[28]
            mostIndex = row[29]

            imagelist = json.loads(imagelist)
            imagelist = [os.path.basename(impath).replace(".jpg","") for impath in imagelist]

            all_coeffs = []
            for filename in imagelist:
                personId, *coeffs = filename.split("_")
                personId = int(personId)
                coeffs = [int(coeff) for coeff in coeffs]
                all_coeffs.append(tuple(coeffs))
            all_coeffs = tuple(all_coeffs)

            if personId not in results_person_coeffs:
                results_person_coeffs[personId] = {}
            if all_coeffs not in results_person_coeffs[personId]:
                results_person_coeffs[personId][all_coeffs] = []

            results_person_coeffs[personId][all_coeffs].append((leastIndex, mostIndex))
    return results_person_coeffs

def calculate_concensus(results):
    for personId in results:
        for coeffs in results[personId]:
            votes = results[personId][coeffs]

            candidate_total = np.zeros(3)
            for vote in votes:
                candidate_score = np.ones(3)
                candidate_score[vote[0]] = 0
                candidate_score[vote[1]] = 2

                candidate_total += candidate_score

            concensus = (np.argmin(candidate_total), np.argmax(candidate_total))

            results[personId][coeffs] = concensus

if __name__ == "__main__":

    mturkResults = r"C:\Drive\FaceAnalysis\Batch_2801527_batch_results.csv"
    results = readMturkResults(mturkResults)

    calculate_concensus(results)

    plt.figure(1)
    plt.title('coeffs')
    plt.xlabel('Coeff 1')
    plt.ylabel('Coeff 2')
    plt.grid(True)
    plt.xlim((-3, 3))
    plt.ylim((-3, 3))

    max_I = 10
    srcFolder = r"C:\Facedata\10k US Adult Faces Database\Publication Friendly 49-Face Database\49 Face Images\*.jpg"
    for personId, impath in enumerate(glob.glob(srcFolder)):
        if personId >= max_I:
            break
        im = cv2.imread(impath)
        filename = os.path.basename(impath)

        landmarks = getLandmarks(im)
        mesh, pose, shape_coeffs, blendshape_coeffs = getMeshFromLandmarks(landmarks, im, num_iterations=300)

        pResults = results[personId]
        origPoint = blendshape_coeffs[:2]

        plt.plot(origPoint[0], origPoint[1], 'ro')
        for coeffs in [((-1, 0), (0, 0), (1, 0)), ((0, -1), (0, 0), (0, 1))]:#pResults:
            coeff_delta = pResults[coeffs]

            scale=0.3
            origPoints = (np.array(coeffs)*scale) + origPoint

            startP = origPoints[coeff_delta[0],:]
            endP = origPoints[coeff_delta[1],:]
            dP = endP - startP

            plt.arrow(startP[0], startP[1], dP[0], dP[1], fc="k", ec="k", head_width=0.05, head_length=0.1)

    plt.show()
