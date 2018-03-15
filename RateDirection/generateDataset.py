import cv2
import os
import eos
import itertools
import random
import numpy as np
from Beautifier.faceFeatures import getLandmarks
from Beautifier.face3D.warpFace3D import warpFace3D
from Beautifier.face3D.faceFeatures3D import BFM_FACEFITTING, SFM_FACEFITTING
from Beautifier.faceCNN.faceFeaturesCNN import getFaceFeaturesCNN
import csv
import json

def create_filename(im_i, coeff_deltas):
    return "{}_{}.jpg".format(im_i, "_".join(["{:d}".format(int(d)) for d in coeff_deltas]))

if __name__ == "__main__":
    import glob

    srcFolder = r"E:\Facedata\10k US Adult Faces Database\Publication Friendly 49-Face Database\49 Face Images_F\*.jpg"
    dstFolder = "./results/"

    with open(os.path.join(dstFolder,'mturk_batch.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename','shape_coeffs_sfm','shape_coeffs_bfm','coeff_is','imageList'])

        max_I = 100
        for i, impath in enumerate(glob.glob(srcFolder)):
            if i >= max_I:
                break
            print(impath)
            im = cv2.imread(impath)
            filename = os.path.basename(impath)

            landmarks = getLandmarks(im)
            mesh_sfm, pose_sfm, facefeatures_sfm, blendshape_coeffs_sfm = SFM_FACEFITTING.getMeshFromLandmarks(landmarks, im, num_iterations=300)
            # mesh_sfm, pose_sfm, facefeatures_sfm, blendshape_coeffs_sfm = SFM_FACEFITTING.getMeshFromLandmarks(landmarks, im, num_iterations=5000)

            facefeatures_bfm = getFaceFeaturesCNN([im])[0:99]
            pose_bfm = BFM_FACEFITTING.getPoseFromShapeCeoffs(landmarks, im, facefeatures_bfm)
            mesh_bfm = BFM_FACEFITTING.getMeshFromShapeCeoffs(facefeatures_bfm)

            num_coeffs = 12
            coeff_deltas = []
            for c in range(num_coeffs):
                coeffs_up = [0] * num_coeffs
                coeffs_dn = [0] * num_coeffs
                coeffs_up[c] = 1
                coeffs_dn[c] = -1
                coeff_deltas.append(coeffs_up)
                coeff_deltas.append(coeffs_dn)

            # for coeff_delta_vec in coeff_deltas:
            #     print(coeff_delta_vec)
            #
            #     new_coeffs_sfm = facefeatures_sfm.copy()
            #     new_coeffs_sfm[:num_coeffs] += np.array(coeff_delta_vec)
            #
            #     new_coeffs_bfm = facefeatures_bfm.copy()
            #     new_coeffs_bfm[:num_coeffs] += np.array(coeff_delta_vec)
            #
            #     mesh_sfm_new = SFM_FACEFITTING.getMeshFromShapeCeoffs(new_coeffs_sfm, blendshape_coeffs_sfm)
            #     mesh_bfm_new = BFM_FACEFITTING.getMeshFromShapeCeoffs(new_coeffs_bfm)
            #
            #
            #     warpedIm_sfm = warpFace3D(im, mesh_sfm, pose_sfm, mesh_sfm_new, accurate=True)
            #     warpedIm_bfm = warpFace3D(im, mesh_bfm, pose_bfm, mesh_bfm_new, accurate=True)
            #
            #
            #     cv2.imshow("warped_sfm", warpedIm_sfm)
            #     cv2.imshow("warped_bfm", warpedIm_bfm)
            #
            #     outFile_sfm = os.path.join(os.path.join(dstFolder, "sfm"), create_filename(i, coeff_delta_vec))
            #     outFile_bfm = os.path.join(os.path.join(dstFolder, "bfm"), create_filename(i, coeff_delta_vec))
            #     cv2.imwrite(outFile_sfm, warpedIm_sfm)
            #     cv2.imwrite(outFile_bfm, warpedIm_bfm)
            #     cv2.waitKey(1)

            host_url = "https://crowdrobotics.org/static/img/mturk/pilot/"
            num_per_HIT = 3
            num_assignments = 6
            num_HITs = int(num_coeffs / num_per_HIT)

            for a in range(num_assignments):
                cIndexs = set(range(num_coeffs))


                for h in range(num_HITs):
                    randIndexs = random.sample(cIndexs, num_per_HIT)
                    cIndexs = cIndexs - set(randIndexs)

                    coeffs = []
                    for cIndex in randIndexs:
                        index = cIndex * 2
                        coeffs.append(coeff_deltas[index])
                        coeffs.append(coeff_deltas[index + 1])

                    faceList = []
                    for coeff_delta_vec in coeffs:
                        outFile = host_url + create_filename(i, coeff_delta_vec)
                        faceList.append(outFile)

                    writer.writerow([filename, json.dumps(facefeatures_sfm), json.dumps(facefeatures_bfm.tolist()), json.dumps(randIndexs), json.dumps(faceList)])