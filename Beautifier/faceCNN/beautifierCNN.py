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
    from Beautifier.face3D.faceFeatures3D import getMeshFromLandmarks, getMeshFromShapeCeoffs, getPoseFromShapeCeoffs
    from Beautifier.face3D.warpFace3D import drawMesh, warpFace3D
    from Beautifier.faceCNN.SFM_2_BFM import BFM_2_SFM

    # ## Loading the Basel Face Model to write the 3D output
    BFM_path = os.path.join(os.environ['CNN_PATH'], 'BaselFaceModel_mod.mat')
    attr_path = os.path.join(os.environ['CNN_PATH'], '04_attributes.mat')
    model = scipy.io.loadmat(BFM_path, squeeze_me=True, struct_as_record=False)
    attributes = scipy.io.loadmat(attr_path, squeeze_me=True, struct_as_record=False)
    model = model["BFM"]
    faces = model.faces - 1

    GENDER = "M"

    dstFolder = "./resultsCNN{}eB/".format(GENDER)

    datasets = [#loadUS10k(type="cnn", gender=GENDER),
                loadRateMe(type="cnn", gender=GENDER)]

    trainX, trainY, pca, gp = datasets[0]

    # imFolder = r"E:\Facedata\RateMe\21_F_423ekl\*.jpg"  # 8.1
    imFolder = r"C:\Users\ellio\PycharmProjects\circlelines\Beautifier\face3D\me\*.jpg"
    # imFolder = r"E:\Facedata\10k US Adult Faces Database\Publication Friendly 49-Face Database\49 Face Images\*.jpg"

    print("begin beautification")
    for i, impath in enumerate([r"E:\Facedata\RateMe\22_F_3g5498\5i7r2x0.jpg"]):#glob.glob(imFolder)):
        print(i)
        im = cv2.imread(impath)
        filename = os.path.basename(impath)

        try:
            landmarks = getLandmarks(im)

            #fit an sfm mesh to the face, we need the pose and blendshapes
            mesh_sfm_orig, pose_sfm_orig, facefeatures_sfm_orig, blendshape_coeffs_sfm = getMeshFromLandmarks(landmarks, im)
            facefeatures_sfm_orig = np.array(facefeatures_sfm_orig)

            #get the BFM facefeatures whioh is more accurate, convert sfm so we can do the morphing
            facefeatures_bfm = getFaceFeaturesCNN([im])
            facefeatures_sfm = BFM_2_SFM(facefeatures_bfm, facefeatures_sfm_orig)
            mesh_sfm = getMeshFromShapeCeoffs(facefeatures_sfm, blendshape_coeffs_sfm)
            pose_sfm = getPoseFromShapeCeoffs(landmarks, im, facefeatures_sfm, blendshape_coeffs_sfm)

            #optimise new BFM facefeatures to be more attractive
            new_facefeatures_bfm = facefeatures_bfm.copy()

            scale = -2
            new_facefeatures_bfm[0:99] = new_facefeatures_bfm[0:99] - (attributes['gender_shape'][0:99] * scale)
            # new_facefeatures_bfm[0:99] = new_facefeatures_bfm[0:99] - (attributes['weight_shape'][0:99] * scale)
            # new_facefeatures_bfm[0:99] = new_facefeatures_bfm[0:99] + (attributes['height_shape'][0:99] * scale)
            # new_facefeatures_bfm[0:99] = new_facefeatures_bfm[0:99] - (attributes['age_shape'][0:99] * scale)
            # new_facefeatures_bfm[0:99] = findBestFeaturesOptimisation(facefeatures_bfm[0:99], gp)

            #convert to SFM features
            new_facefeatures_sfm = BFM_2_SFM(new_facefeatures_bfm, facefeatures_sfm)
            new_mesh_sfm = getMeshFromShapeCeoffs(new_facefeatures_sfm, blendshape_coeffs_sfm)

            warpedIm = warpFace3D(im, mesh_sfm, pose_sfm, new_mesh_sfm, accurate=True)
            cv2.imshow("orig", im)
            cv2.imshow("bfm", warpedIm)
            cv2.imshow("mesh_orig", drawMesh(im, mesh_sfm_orig, pose_sfm_orig))
            cv2.imshow("mesh_bfm_orig_pose", drawMesh(im, mesh_sfm, pose_sfm_orig))
            cv2.imshow("mesh_bfm", drawMesh(im, mesh_sfm, pose_sfm))
            cv2.imshow("mesh_new", drawMesh(im, new_mesh_sfm, pose_sfm))

            cv2.imwrite(os.path.join(dstFolder, "{}_0.jpg".format(filename)), im)
            cv2.imwrite(os.path.join(dstFolder, "{}_1.jpg".format(filename)), warpedIm)
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