import cv2
import os
import eos
import numpy as np
from Beautifier.faceFeatures import getLandmarks
from Beautifier.face3D.warpFace3D import warpFace3D
from Beautifier.face3D.faceFeatures3D import getMeshFromLandmarks, model, blendshapes

if __name__ == "__main__":
    import glob

    srcFolder = r"E:\Facedata\10k US Adult Faces Database\Publication Friendly 49-Face Database\49 Face Images\*.jpg"
    dstFolder = "./results/"

    for i, impath in enumerate([r"C:\Users\ellio\Desktop\girl-people-landscape-sun-38554.jpeg"]):#glob.glob(srcFolder)):
        print(i)
        im = cv2.imread(impath)
        filename = os.path.basename(impath)

        landmarks = getLandmarks(im)
        mesh, pose, shape_coeffs, blendshape_coeffs = getMeshFromLandmarks(landmarks, im)

        newim = np.zeros((im.shape[0] * 5, im.shape[1] * 5, im.shape[2]), np.uint8)
        count = 0
        for coeff_i in range(10):
            for j, coeff_delta in enumerate(np.linspace(-2, 2, 5)):
                new_coeffs = list(shape_coeffs)
                new_coeffs[coeff_i] += coeff_delta

                newMesh = eos.morphablemodel.draw_sample(model, blendshapes, new_coeffs, blendshape_coeffs, [])

                warpedIm = warpFace3D(im, mesh, pose, newMesh)
                cv2.imshow("warped", warpedIm)
                outFile = os.path.join(dstFolder, "{}_{}_{}.jpg".format(i,coeff_i,j))
                cv2.imwrite(outFile, warpedIm)
                cv2.waitKey(1)