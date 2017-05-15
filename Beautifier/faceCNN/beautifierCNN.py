import numpy as np
import cv2
import os
import scipy

scriptFolder = os.path.dirname(os.path.realpath(__file__))

def findBestFeaturesOptimisation(featuresCNN, gp):
    print("finding optimal face features optimisation")

    def GPCostFunction(features):
        y_pred = gp.predict([features])
        return -y_pred

    bounds = np.zeros((featuresCNN.shape[0],2))
    bounds[:, 0] = -200#featuresCNN - 0.5
    bounds[:, 1] = 200#featuresCNN + 0.5

    optimalNewFaceFeatures = scipy.optimize.minimize(GPCostFunction, featuresCNN, bounds=bounds, method='SLSQP', options={"maxiter": 5, "eps": 0.001})
    return optimalNewFaceFeatures.x

if __name__ == "__main__":
    import eos
    import glob
    from US10k.US10k import loadUS10k
    from RateMe.RateMe import loadRateMe
    import Beautifier.faceCNN.utils as utils
    import scipy.io
    from sklearn.linear_model import LinearRegression
    from Beautifier.faceCNN.faceFeaturesCNN import getFaceFeaturesCNN
    from Beautifier.faceFeatures import getLandmarks
    from Beautifier.face3D.faceFeatures3D import model_bfm, landmark_ids, landmark_mapper_bfm, getMeshFromLandmarks, model as model_sfm, landmark_mapper as landmark_mapper_sfm
    from Beautifier.face3D.warpFace3D import drawMesh, warpFace3D

    # ## Loading the Basel Face Model to write the 3D output
    BFM_path = os.path.join(os.environ['CNN_PATH'], 'BaselFaceModel_mod.mat')
    attr_path = os.path.join(os.environ['CNN_PATH'], '04_attributes.mat')
    model = scipy.io.loadmat(BFM_path, squeeze_me=True, struct_as_record=False)
    attributes = scipy.io.loadmat(attr_path, squeeze_me=True, struct_as_record=False)
    model = model["BFM"]
    faces = model.faces - 1

    GENDER = "M"

    dstFolder = "./resultsCNN{}/".format(GENDER)

    datasets = [#loadUS10k(type="cnn", gender=GENDER),
                loadRateMe(type="cnn", gender=GENDER)]

    trainX, trainY, pca, gp = datasets[0]

    reg = LinearRegression()
    reg.fit(trainX[:,0:99], trainY)

    print("begin beautification")
    for i, impath in enumerate([r"C:\Users\ellio\PycharmProjects\circlelines\Beautifier\face3D\me\8.jpg"]):#enumerate(glob.glob(r"E:\Facedata\10k US Adult Faces Database\Publication Friendly 49-Face Database\49 Face Images\*.jpg")):
        print(i)
        im = cv2.imread(impath)
        filename = os.path.basename(impath)

        trainX, trainY, pca, gp = datasets[0]

        try:
            landmarks = getLandmarks(im)

            # for j in [5, 50, 100, 200, 300]:
            #     mesh_sfm, pose_sfm, shape_coeffs_sfm, blendshape_coeffs_sfm = getMeshFromLandmarks(landmarks, im, num_iterations=j)
            # # pose_sfm = eos.fitting.fit_pose(model_sfm, landmarks, landmark_ids, landmark_mapper_sfm, im.shape[1], im.shape[0], shape_coeffs_sfm)
            #     drawIm = drawMesh(im, mesh_sfm, pose_sfm)
            #     cv2.imshow("sfm_{}".format(j), drawIm)
            # cv2.waitKey(-1)


            facefeatures_bfm = getFaceFeaturesCNN([im])
            pose_bfm = eos.fitting.fit_pose(model_bfm, landmarks, landmark_ids, landmark_mapper_bfm, im.shape[1], im.shape[0], facefeatures_bfm)
            mesh_bfm = eos.morphablemodel.draw_sample(model_bfm, [], facefeatures_bfm, [], [])

            drawIm = drawMesh(im, mesh_bfm, pose_bfm)
            cv2.imshow("bfm", drawIm)
            cv2.waitKey(-1)


            newfacefeatures = facefeatures_bfm.copy()

            scale = 10
            # newfacefeatures[0:99] = newfacefeatures[0:99] - (reg.coef_[0:99] * scale)
            # newfacefeatures[0:99] = newfacefeatures[0:99] - (attributes['gender_shape'][0:99] * scale)
            newfacefeatures[0:99] = newfacefeatures[0:99] - (attributes['weight_shape'][0:99] * scale)
            # newfacefeatures[0:99] = newfacefeatures[0:99] + (attributes['height_shape'][0:99] * scale)
            # newfacefeatures[0:99] = newfacefeatures[0:99] - (attributes['age_shape'][0:99] * scale)


            # newfacefeatures[0:99] = findBestFeaturesOptimisation(facefeatures_bfm[0:99], gp)

            mesh_bfm_new = eos.morphablemodel.draw_sample(model_bfm, [], newfacefeatures, [], [])

            warpedIm = warpFace3D(im, mesh_bfm, pose_bfm, mesh_bfm_new)
            cv2.imshow("orig", im)
            cv2.imshow("bfm", warpedIm)
            cv2.waitKey(-1)

            # outfileOLD = os.path.join(dstFolder, "%s_OLD.ply" % filename)
            # outfileNEW = os.path.join(dstFolder, "%s_NEW.ply" % filename)
            # S, T = utils.projectBackBFM(model, facefeatures)
            # utils.write_ply(outfileOLD, S, T, faces)
            #
            # S, T = utils.projectBackBFM(model, newfacefeatures)
            # utils.write_ply(outfileNEW, S, T, faces)
        except Exception as e:
            print(e)