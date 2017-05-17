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
            leastIndex = int(row[28])
            mostIndex = int(row[29])

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

def readMturkResults2(filepath):
    results_by_image = {}
    shape_coeffs_by_image = {}

    with open(filepath, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)

        for i, row in enumerate(reader):
            if i == 0:
                continue

            workerId = row[15]
            filename = row[27]
            shape_coeffs = json.loads(row[28])
            imagelist = json.loads(row[29])
            chosenFaces = np.array(row[30].split(","), dtype=np.int32)
            stages = np.array(row[31].split(","), dtype=np.int32)
            swapCount = row[32]

            imagelist = [os.path.basename(impath).replace(".jpg", "") for impath in imagelist]

            all_coeffs = []
            for imagename in imagelist:
                personId, *coeffs = imagename.split("_")
                personId = int(personId)
                coeffs = [int(coeff) for coeff in coeffs]
                all_coeffs.append(tuple(coeffs))
            all_coeffs = tuple(all_coeffs)

            if filename not in results_by_image:
                results_by_image[filename] = []

            results_by_image[filename].append((workerId, chosenFaces, stages, swapCount, all_coeffs))
            shape_coeffs_by_image[filename] = shape_coeffs
    return results_by_image, shape_coeffs_by_image

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

def calculate_concensus2(results_by_image):
    vector_by_image = {}
    all_contradictions = {}
    for filename in results_by_image:
        votes = results_by_image[filename]

        avg_vector = np.zeros(2)
        contradictions = []
        for vote in votes:
            chosenFaces = vote[1]
            stages = vote[2]

            vector = (chosenFaces * 2) - 1
            vector = vector[:2]

            avg_vector += vector

            if chosenFaces[0] != chosenFaces[2]:
                contradictions.append(vote)

        avg_vector /= len(votes)

        vector_by_image[filename] = avg_vector
        all_contradictions[filename] = contradictions

        print("{}: {}/{}".format(filename, len(contradictions),len(votes)))

    return vector_by_image

if __name__ == "__main__":

    # mturkResults = r"E:\Drive\FaceAnalysis\Batch_2801527_batch_results.csv"
    mturkResults = r"E:\Drive\FaceAnalysis\Batch_2806769_batch_results.csv"

    results_by_image, shape_coeffs_by_image = readMturkResults2(mturkResults)

    vector_by_image = calculate_concensus2(results_by_image)

    plt.figure(1)
    plt.title('coeffs')
    plt.xlabel('Coeff 1')
    plt.ylabel('Coeff 2')
    plt.grid(True)
    plt.xlim((-3, 3))
    plt.ylim((-3, 3))

    vectors = []
    for filename in vector_by_image:
        shape_coeffs = shape_coeffs_by_image[filename]
        vector = vector_by_image[filename]

        origPoint = shape_coeffs[:2]

        plt.gca().annotate("", xy=origPoint+vector, xytext=origPoint, arrowprops=dict(arrowstyle="->", color='r'), color='r')

        vectors.append((origPoint, vector))

    # plt.show()
    num_arrows = 40
    X, Y = np.meshgrid(np.linspace(-1.5, 1.5, num_arrows), np.linspace(-1.5, 1.5, num_arrows))
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    P = 6

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):

            p = np.array((X[i][j], Y[i][j]))

            numerator = np.zeros_like(p)
            denominator = 0
            for influence in vectors:
                squared_distance = np.sum(np.square(p - influence[0]))
                divisor = np.power(squared_distance, P/2)

                numerator += influence[1] / divisor
                denominator += 1/divisor


            v = numerator/denominator

            U[i][j] = v[0]
            V[i][j] = v[1]

    plt.figure()
    plt.title('Arrows scale with plot width, not view')
    Q = plt.quiver(X, Y, U, V, units='width')
    qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
                       coordinates='figure')

    plt.show()
