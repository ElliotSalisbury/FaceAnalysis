import cv2
import os
import eos
import itertools
import numpy as np
from Beautifier.faceFeatures import getLandmarks
from Beautifier.face3D.warpFace3D import warpFace3D
from Beautifier.face3D.faceFeatures3D import getMeshFromLandmarks, model, blendshapes
import csv
import json

if __name__ == "__main__":
    import glob

    srcFolder = r"E:\Facedata\10k US Adult Faces Database\Publication Friendly 49-Face Database\49 Face Images\*.jpg"
    dstFolder = "./results/"

    with open(os.path.join(dstFolder,'mturk_batch.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['imageList'])

        for impath in [r"C:\Users\ellio\Desktop\girl-people-landscape-sun-38554.jpeg"]:#glob.glob(srcFolder)):
            im = cv2.imread(impath)
            filename = os.path.basename(impath)

            landmarks = getLandmarks(im)
            mesh, pose, shape_coeffs, blendshape_coeffs = getMeshFromLandmarks(landmarks, im, num_iterations=300)

            newim = np.zeros((im.shape[0] * 5, im.shape[1] * 5, im.shape[2]), np.uint8)

            coeff_deltas = np.linspace(-1, 1, 3)
            n_coeffs = 2

            for coeff_delta_vec in itertools.product(coeff_deltas, repeat=n_coeffs):
                coeff_delta_vec = np.array(coeff_delta_vec)
                new_coeffs = shape_coeffs.copy()
                new_coeffs[:coeff_delta_vec.shape[0]] += coeff_delta_vec

                newMesh = eos.morphablemodel.draw_sample(model, blendshapes, new_coeffs, blendshape_coeffs, [])

                warpedIm = warpFace3D(im, mesh, pose, newMesh)
                cv2.imshow("warped", warpedIm)
                outFile = os.path.join(dstFolder, "{}.jpg".format(coeff_delta_vec))
                cv2.imwrite(outFile, warpedIm)
                cv2.waitKey(1)

            mturk_experiments = []
            for coeff_0 in coeff_deltas:
                images = []
                for coeff_1 in coeff_deltas:
                    coeff_delta_vec = np.array([coeff_0, coeff_1])
                    outFile = os.path.join(dstFolder, "http://objctify.me/static/img/mturk/pilot/{}.jpg".format(coeff_delta_vec))
                    images.append(outFile)
                mturk_experiments.append(images)

            for coeff_1 in coeff_deltas:
                images = []
                for coeff_0 in coeff_deltas:
                    coeff_delta_vec = np.array([coeff_0, coeff_1])
                    outFile = os.path.join(dstFolder, "http://objctify.me/static/img/mturk/pilot/{}.jpg".format(coeff_delta_vec))
                    images.append(outFile)
                mturk_experiments.append(images)

            for exp in mturk_experiments:
                writer.writerow([json.dumps(exp)])