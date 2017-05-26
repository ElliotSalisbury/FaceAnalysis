import os
import scipy.io
import numpy as np
import cv2
import Beautifier.faceCNN.utils as utils
import shapely.geometry
from Beautifier.face3D.faceFeatures3D import getMeshFromLandmarks, getMeshFromShapeCeoffs, landmarks_2_vert_indices, landmarks_2_vert_indices_bfm

BFM_path = os.path.join(os.environ['CNN_PATH'], 'BaselFaceModel_mod.mat')
# ## Loading the Basel Face Model to write the 3D output
BFM_MODEL = scipy.io.loadmat(BFM_path, squeeze_me=True, struct_as_record=False)
BFM_MODEL = BFM_MODEL["BFM"]
BFM_FACES = BFM_MODEL.faces - 1

def BFM_2_SFM(bfm_shape_coeffs, sfm_shape_coeffs_guess):
    bfm_verts, bfm_texts = utils.projectBackBFM(BFM_MODEL, bfm_shape_coeffs)

    comparison_landmarks = []
    for i in range(68):
        sfm_v_i = landmarks_2_vert_indices[i]
        bfm_v_i = landmarks_2_vert_indices_bfm[i]
        if sfm_v_i != -1 and bfm_v_i != -1:
            comparison_landmarks.append((sfm_v_i, bfm_v_i))
    comparison_landmarks.append((1947, 39032))
    # comparison_landmarks.append((1998, 14383))
    # comparison_landmarks.append((2127, 2385))
    comparison_landmarks.append((2110, 43783))
    comparison_landmarks = np.array(comparison_landmarks)

    # points_bfm = []
    # for vert in bfm_verts:
    #     points_bfm.append(shapely.geometry.Point(vert))
    # points_bfm = shapely.geometry.MultiPoint(points_bfm)
    #
    # bfm_max_z = np.max(S[:, 2])
    NOSE_LANDMARK = 30
    bfm_nose_v_i = landmarks_2_vert_indices_bfm[NOSE_LANDMARK]
    sfm_nose_v_i = landmarks_2_vert_indices[NOSE_LANDMARK]
    max_nose_v = bfm_verts[bfm_nose_v_i]

    iterCount = 0
    def mesh_distance(sfm_shape):
        nonlocal iterCount
        sfm_mesh = getMeshFromShapeCeoffs(sfm_shape)

        sfm_verts = np.array(sfm_mesh.vertices)[:, :3]
        sfm_verts = sfm_verts + np.array([0,0,( max_nose_v - sfm_verts[sfm_nose_v_i] )[2]])
        # points_sfm = []
        # for vert in verts_sfm:
        #     points_sfm.append(shapely.geometry.Point(vert))

        # distance_score = 0
        # for p_sfm in points_sfm:
        #     dist = points_bfm.distance(p_sfm)
        #     distance_score += dist
        deltas = sfm_verts[comparison_landmarks[:,0]] - bfm_verts[comparison_landmarks[:,1]]
        distance = np.square(np.linalg.norm(deltas, axis=1))
        distance_score = np.sum(distance)

        scale = 20
        regularisation = (scale/sfm_shape.shape[0]) * np.sum(np.square(sfm_shape))
        distance_score += regularisation
        iterCount += 1
        # if iterCount % 100 == 0:
        # print("%i - %0.2f" % (iterCount, distance_score))

        return distance_score

    small_size = 12
    sfm_shape_coeffs_guess_small = sfm_shape_coeffs_guess[:small_size]

    bounds = np.zeros((sfm_shape_coeffs_guess_small.shape[0], 2))
    bounds[:, 0] = -5
    bounds[:, 1] = 5

    sfm_shape_coeffs_opt = scipy.optimize.minimize(mesh_distance, sfm_shape_coeffs_guess_small, bounds=bounds, method='SLSQP', options={"maxiter": 50, "eps": 0.001})
    sfm_shape_coeffs_opt = sfm_shape_coeffs_opt.x

    # r = sfm_shape_coeffs_guess.copy()
    # r[:small_size] = sfm_shape_coeffs_opt

    return sfm_shape_coeffs_opt

if __name__ == "__main__":
    from Beautifier.faceCNN.faceFeaturesCNN import getFaceFeaturesCNN
    from Beautifier.faceFeatures import getLandmarks

    dstFolder = "./results/"
    impath = r"E:\Facedata\RateMe\21_F_423ekl\HeC768w.jpg"
    impath = r"C:\Users\ellio\PycharmProjects\circlelines\Beautifier\face3D\me\0132.jpg"
    # impath = r"E:\Facedata\RateMe\18_M_2wbhfw\n5DesJ4.jpg"
    im = cv2.imread(impath)
    filename = os.path.basename(impath)

    outfile_bfm = os.path.join(dstFolder, "%s_bfm.ply" % filename)
    outfile_sfm_orig = os.path.join(dstFolder, "%s_sfm_orig.ply" % filename)
    outfile_com_orig = os.path.join(dstFolder, "%s_com_orig.ply" % filename)
    outfile_sfm_opt = os.path.join(dstFolder, "%s_sfm_opt.ply" % filename)
    outfile_com_opt = os.path.join(dstFolder, "%s_com_opt.ply" % filename)

    #bfm model
    bfm_features = getFaceFeaturesCNN([im])
    S, T = utils.projectBackBFM(BFM_MODEL, bfm_features)
    utils.write_ply(outfile_bfm, S, T, BFM_FACES)
    print(bfm_features)

    # sfm model
    landmarks = getLandmarks(im)
    sfm_mesh, pose, sfm_features, blendshape_coeffs = getMeshFromLandmarks(landmarks, im)
    sfm_features = np.array(sfm_features)
    print(sfm_features)

    verts_sfm = np.array(sfm_mesh.vertices)[:, :3]
    verts_sfm = verts_sfm + np.array([0, 0, np.max(S[:, 2]) - np.max(verts_sfm[:, 2])])
    text_sfm = np.ones_like(verts_sfm) * 128
    faces_sfm = np.array(sfm_mesh.tvi)
    utils.write_ply(outfile_sfm_orig, verts_sfm, text_sfm, faces_sfm)

    # combine
    com_verts = np.vstack([verts_sfm, S])
    com_text = np.vstack([text_sfm, T])
    com_faces = np.vstack([faces_sfm, BFM_FACES + verts_sfm.shape[0]])
    utils.write_ply(outfile_com_orig, com_verts, com_text, com_faces)



    #optimise
    new_sfm_shape = BFM_2_SFM(bfm_features, sfm_features)
    print(new_sfm_shape)

    # sfm model
    sfm_mesh = getMeshFromShapeCeoffs(new_sfm_shape)

    verts_sfm = np.array(sfm_mesh.vertices)[:, :3]
    verts_sfm = verts_sfm + np.array([0, 0, np.max(S[:, 2]) - np.max(verts_sfm[:, 2])])
    text_sfm = np.ones_like(verts_sfm) * 128
    faces_sfm = np.array(sfm_mesh.tvi)
    utils.write_ply(outfile_sfm_opt, verts_sfm, text_sfm, faces_sfm)

    # combine
    com_verts = np.vstack([verts_sfm, S])
    com_text = np.vstack([text_sfm, T])
    com_faces = np.vstack([faces_sfm, BFM_FACES + verts_sfm.shape[0]])
    utils.write_ply(outfile_com_opt, com_verts, com_text, com_faces)

    print("done")