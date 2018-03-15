import numpy as np
import cv2
import os
import scipy
import json
from Beautifier.face3D.faceFeatures3D import BFM_FACEFITTING
from Beautifier.face3D.warpFace3D import ALL_FACE_LANDMARKS, projectMeshTo2D
from Beautifier.faceCNN.faceFeaturesCNN import getFaceFeaturesCNN
from Beautifier.faceFeatures import getLandmarks

scriptFolder = os.path.dirname(os.path.realpath(__file__))

def findBestFeaturesOptimisation(featuresCNN, predictor):
    print("finding optimal face features optimisation")

    def GPCostFunction(features):
        y_pred = predictor.predict([features])
        return -y_pred

    bounds = np.zeros((featuresCNN.shape[0],2))
    bounds[:, 0] = -200#featuresCNN - 0.5
    bounds[:, 1] = 200#featuresCNN + 0.5

    optimalNewFaceFeatures = scipy.optimize.minimize(GPCostFunction, featuresCNN, bounds=bounds, method='SLSQP', options={"maxiter": 5, "eps": 0.001})
    return optimalNewFaceFeatures.x

def getWebResults(im, predictor):
    facefeatures_bfm = getFaceFeaturesCNN([im])[0:99]

    rating = predictor.predict([facefeatures_bfm])[0]

    # new_facefeatures_bfm = facefeatures_bfm.copy()
    new_facefeatures_bfm = findBestFeaturesOptimisation(facefeatures_bfm, predictor)

    landmarks = getLandmarks(im)
    pose_bfm = BFM_FACEFITTING.getPoseFromShapeCeoffs(landmarks, im, facefeatures_bfm)
    mesh_bfm = BFM_FACEFITTING.getMeshFromShapeCeoffs(facefeatures_bfm)
    modelview, projection, viewport, indexs = decomposePose(mesh_bfm, pose_bfm, im)

    results = {}
    results["facefeatures"] = facefeatures_bfm.tolist()
    results["rating"] = rating
    results["new_facefeatures"] = new_facefeatures_bfm.tolist()
    results["modelview"] = modelview
    results["projection"] = projection
    results["viewport"] = viewport
    results["indexs"] = indexs

    return results


def decomposePose(mesh, pose, im):
    modelview = np.matrix(pose.get_modelview())
    proj = np.matrix(pose.get_projection())
    viewport = np.array([0, im.shape[0], im.shape[1], -im.shape[0]])

    modelview = modelview.tolist()
    projection = proj.tolist()
    viewport = viewport.tolist()

    ALL_FACE_MESH_VERTS = BFM_FACEFITTING.landmarks_2_vert_indices[ALL_FACE_LANDMARKS]
    ALL_FACE_MESH_VERTS = np.delete(ALL_FACE_MESH_VERTS, np.where(ALL_FACE_MESH_VERTS == -1)).tolist()
    verts2d = projectMeshTo2D(mesh, pose, im)
    convexHullIndexs = cv2.convexHull(verts2d.astype(np.float32), returnPoints=False)
    warpPointIndexs = convexHullIndexs.flatten().tolist() + ALL_FACE_MESH_VERTS
    indexs = warpPointIndexs

    return modelview, projection, viewport, indexs

def poseToJS(mesh, pose, im):
    modelview, projection, viewport, indexs = decomposePose(mesh, pose, im)

    js_modelview = "var modelview = math.matrix({});".format(json.dumps(modelview))
    js_proj = "var projection = math.matrix({});".format(json.dumps(projection))
    js_viewport = "var viewport = {};".format(json.dumps(viewport))
    js_indexs = "var indexs = {};".format(json.dumps(indexs))

    print(js_modelview)
    print(js_proj)
    print(js_viewport)
    print(js_indexs)

if __name__ == "__main__":
    # import glob
    from RateMe.RateMe import loadRateMe
    from Beautifier.face3D.faceFeatures3D import SFM_FACEFITTING, createTextureMap
    # import Beautifier.faceCNN.utils as utils
    # import scipy.io
    # from sklearn.linear_model import LinearRegression
    # from Beautifier.faceFeatures import getLandmarks
    from Beautifier.face3D.warpFace3D import drawMesh, warpFace3D
    from Beautifier.faceCNN.SFM_2_BFM import BFM_2_SFM

    # ## Loading the Basel Face Model to write the 3D output
    # BFM_path = os.path.join(os.environ['CNN_PATH'], 'BaselFaceModel_mod.mat')
    # attr_path = os.path.join(os.environ['CNN_PATH'], '04_attributes.mat')
    # model = scipy.io.loadmat(BFM_path, squeeze_me=True, struct_as_record=False)
    # attributes = scipy.io.loadmat(attr_path, squeeze_me=True, struct_as_record=False)
    # model = model["BFM"]
    # faces = model.faces - 1

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
            mesh_sfm_orig, pose_sfm_orig, facefeatures_sfm_orig, blendshape_coeffs_sfm = SFM_FACEFITTING.getMeshFromLandmarks(landmarks, im, num_iterations=50)
            mesh_sfm_orig_2, pose_sfm_orig_2, facefeatures_sfm_orig_2, blendshape_coeffs_sfm_2 = SFM_FACEFITTING.getMeshFromLandmarks(landmarks, im, num_iterations=5000)
            mesh_sfm_orig_3, pose_sfm_orig_3, facefeatures_sfm_orig_3, blendshape_coeffs_sfm_3 = SFM_FACEFITTING.getMeshFromLandmarks(landmarks, im, num_iterations=6000)
            facefeatures_sfm_orig = np.array(facefeatures_sfm_orig)

            #get the BFM facefeatures whioh is more accurate
            facefeatures_bfm = getFaceFeaturesCNN([im])
            pose_bfm = BFM_FACEFITTING.getPoseFromShapeCeoffs(landmarks, im, facefeatures_bfm)
            mesh_bfm = BFM_FACEFITTING.getMeshFromShapeCeoffs(facefeatures_bfm)
            poseToJS(mesh_bfm, pose_bfm, im)

            # convert sfm so we can do the morphing
            facefeatures_sfm = BFM_2_SFM(facefeatures_bfm, facefeatures_sfm_orig)
            mesh_sfm = SFM_FACEFITTING.getMeshFromShapeCeoffs(facefeatures_sfm, blendshape_coeffs_sfm)
            pose_sfm = SFM_FACEFITTING.getPoseFromShapeCeoffs(landmarks, im, facefeatures_sfm, blendshape_coeffs_sfm)

            isomap = createTextureMap(mesh_bfm, pose_bfm, im)
            cv2.imwrite("isomap.jpg", isomap)

            #optimise new BFM facefeatures to be more attractive
            new_facefeatures_bfm = facefeatures_bfm.copy()

            # scale = 5
            # new_facefeatures_bfm[0:99] = new_facefeatures_bfm[0:99] - (attributes['gender_shape'][0:99] * scale)
            # new_facefeatures_bfm[0:99] += (attributes['weight_shape'][0:99] * 40 * scale)
            # new_facefeatures_bfm[0:99] = new_facefeatures_bfm[0:99] + (attributes['height_shape'][0:99] * scale)
            # new_facefeatures_bfm[0:99] = new_facefeatures_bfm[0:99] - (attributes['age_shape'][0:99] * scale)
            new_facefeatures_bfm[0:99] = findBestFeaturesOptimisation(facefeatures_bfm[0:99], gp)

            #convert to SFM features
            new_mesh_bfm = BFM_FACEFITTING.getMeshFromShapeCeoffs(new_facefeatures_bfm)
            new_facefeatures_sfm = BFM_2_SFM(new_facefeatures_bfm, facefeatures_sfm)
            new_mesh_sfm = SFM_FACEFITTING.getMeshFromShapeCeoffs(new_facefeatures_sfm, blendshape_coeffs_sfm)

            warpedIm_bfm = warpFace3D(im, mesh_bfm, pose_bfm, new_mesh_bfm, accurate=True)
            warpedIm_sfm = warpFace3D(im, mesh_sfm, pose_sfm, new_mesh_sfm, accurate=True)
            cv2.imshow("orig", im)
            cv2.imshow("warped_bfm", warpedIm_bfm)
            cv2.imshow("warped_sfm", warpedIm_sfm)
            cv2.imshow("mesh_bfm_orig", drawMesh(im, mesh_bfm, pose_bfm))
            cv2.imshow("mesh_sfm_orig", drawMesh(im, mesh_sfm_orig, pose_sfm_orig))
            cv2.imshow("mesh_bfm_orig_2_sfm", drawMesh(im, mesh_sfm, pose_sfm))

            cv2.imshow("mesh_bfm_opt", drawMesh(im, new_mesh_bfm, pose_bfm))
            cv2.imshow("mesh_bfm_opt_2_sfm", drawMesh(im, new_mesh_sfm, pose_sfm))

            # cv2.imwrite(os.path.join(dstFolder, "{}_0.jpg".format(filename)), im)
            # cv2.imwrite(os.path.join(dstFolder, "{}_1.jpg".format(filename)), warpedIm)
            cv2.waitKey(-1)

            # outfileOLD = os.path.join(dstFolder, "%s_OLD.ply" % filename)
            # outfileNEW = os.path.join(dstFolder, "%s_NEW.ply" % filename)
            # S, T = utils.projectBackBFM(model, facefeatures)
            # utils.write_ply(outfileOLD, S, T, faces)
            #
            # S, T = utils.projectBackBFM(model, newfacefeatures)
            # utils.write_ply(outfileNEW, S, T, faces)
        except Exception as e:
            raise e