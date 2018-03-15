import csv
import collections
import json
import os
import cv2
import matplotlib.pyplot as plt
import glob
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
            shape_coeffs = np.array(json.loads(row[28]))
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
    c_count = sum([len(all_contradictions[f]) for f in all_contradictions])
    t_count = sum([len(results_by_image[f]) for f in results_by_image])
    print("total: {}/{}".format(c_count, t_count))
    return vector_by_image

def vectorAtPos(vectors, pos, P=6):
    numerator = np.zeros_like(pos)
    denominator = 0
    for influence in vectors:
        squared_distance = np.sum(np.square(pos - influence[0]))
        divisor = np.power(squared_distance, P / 2)

        numerator += influence[1] / divisor
        denominator += 1 / divisor

    v = numerator / denominator
    return v

if __name__ == "__main__":
    from Beautifier.faceFeatures import getLandmarks
    from Beautifier.face3D.warpFace3D import warpFace3D
    from Beautifier.face3D.faceFeatures3D import SFM_FACEFITTING, BFM_FACEFITTING
    from Beautifier.faceCNN.faceFeaturesCNN import getFaceFeaturesCNN

    # mturkResults = r"E:\Drive\FaceAnalysis\Batch_2801527_batch_results.csv"
    # mturkResults = r"E:\Drive\FaceAnalysis\Batch_2806769_batch_results.csv"
    mturkResults = r"E:\Facedata\RateMeDirection\Batch_2941367_batch_results.csv"

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
    plt_range = 2
    X, Y = np.meshgrid(np.linspace(-plt_range, plt_range, num_arrows), np.linspace(-plt_range, plt_range, num_arrows))
    U = np.zeros_like(X)
    V = np.zeros_like(Y)



    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            p = np.array((X[i][j], Y[i][j]))
            v = vectorAtPos(vectors, p)

            U[i][j] = v[0]
            V[i][j] = v[1]

    plt.figure()
    plt.title('Arrows scale with plot width, not view')
    Q = plt.quiver(X, Y, U, V, units='width')
    qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
                       coordinates='figure')

    plt.draw()

    # from RateMe.RateMe import loadRateMe
    # trainX, trainY, pca, gp = loadRateMe(type="3d", gender="F")
    # X, Y = np.meshgrid(np.linspace(-plt_range, plt_range, num_arrows), np.linspace(-plt_range, plt_range, num_arrows))
    #
    # Xf = X.flatten()
    #
    # samples = np.zeros((Xf.shape[0],63))
    # samples[:, 0] = Xf
    # samples[:, 1] = Y.flatten()
    # y_preds = gp.predict(samples)
    # y_preds = y_preds.reshape(X.shape)
    #
    # gradientX, gradientY = np.gradient(y_preds)

    # plt.figure()
    # plt.title('svms gradient')
    # Q = plt.quiver(X, Y, gradientX, gradientY, units='width')#
    # plt.show()





    # impath = r"E:\Facedata\RateMe\21_F_423ekl\HeC768w.jpg"
    # impath = r"E:\Facedata\RateMe\21_F_42mxvl\fWIPj9e.jpg"
    impath = r"E:\Facedata\RateMe\21_M_5mx3xt\a2Odd4w.jpg"
    im = cv2.imread(impath)
    landmarks = getLandmarks(im)

    # mesh_sfm, pose_sfm, shape_coeffs_sfm, blendshape_coeffs_sfm = SFM_FACEFITTING.getMeshFromLandmarks(landmarks, im, num_iterations=300)
    shape_coeffs_bfm = getFaceFeaturesCNN([im])[0:99]
    pose_bfm = BFM_FACEFITTING.getPoseFromShapeCeoffs(landmarks, im, shape_coeffs_bfm)
    mesh_bfm = BFM_FACEFITTING.getMeshFromShapeCeoffs(shape_coeffs_bfm)

    # new_shape_coeffs = np.array(shape_coeffs_sfm).copy()
    new_shape_coeffs = np.array(shape_coeffs_bfm).copy()
    distance_moved = 0
    target_distance = 2
    poss = []
    while distance_moved < target_distance:
        pos = new_shape_coeffs[:2]

        v = vectorAtPos(vectors, pos)

        v *= 0.1

        distance_moved += np.linalg.norm(v)
        pos += v
        new_shape_coeffs[:2] = pos
        poss.append(pos.copy())
    poss = np.array(poss)
    plt.plot(poss[:,0],poss[:,1],'r-')

    # newMesh_sfm = SFM_FACEFITTING.getMeshFromShapeCeoffs(new_shape_coeffs, blendshape_coeffs_sfm)
    newMesh_bfm = BFM_FACEFITTING.getMeshFromShapeCeoffs(new_shape_coeffs)

    # warpedIm = warpFace3D(im, mesh_sfm, pose_sfm, newMesh_sfm, accurate=True)
    warpedIm = warpFace3D(im, mesh_bfm, pose_bfm, newMesh_bfm, accurate=True)
    cv2.imshow("orig", im)
    cv2.imshow("warped", warpedIm)
    # cv2.waitKey(1)
    cv2.waitKey(1)
    plt.show()
    cv2.waitKey(-1)