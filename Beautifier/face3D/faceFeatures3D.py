import eos
import numpy as np
import os
import cv2
from Beautifier.faceFeatures import getLandmarks, faceLines

EOS_SHARE_PATH = os.environ['EOS_DATA_PATH']

landmark_ids = list(map(str, range(1, 69)))  # generates the numbers 1 to 68, as strings
model = eos.morphablemodel.load_model(os.path.join(EOS_SHARE_PATH,"sfm_shape_3448.bin"))
blendshapes = eos.morphablemodel.load_blendshapes(os.path.join(EOS_SHARE_PATH,"expression_blendshapes_3448.bin"))
landmark_mapper = eos.core.LandmarkMapper(os.path.join(EOS_SHARE_PATH,"ibug2did.txt"))
edge_topology = eos.morphablemodel.load_edge_topology(os.path.join(EOS_SHARE_PATH,"sfm_3448_edge_topology.json"))
contour_landmarks = eos.fitting.ContourLandmarks.load(os.path.join(EOS_SHARE_PATH,"ibug2did.txt"))
model_contour = eos.fitting.ModelContour.load(os.path.join(EOS_SHARE_PATH,"model_contours.json"))

vertIndices = [landmark_mapper.convert(str(l)) for l in range(69)]
vertIndices = [int(i) if i else -1 for i in vertIndices]

newFaceLines = []
for line in faceLines:
    if vertIndices[line[0]] == -1 or vertIndices[line[1]] == -1:
        continue
    newFaceLines.append(line)
newFaceLines = np.array(newFaceLines)

def getMeshFromLandmarks(landmarks, im, num_iterations=5, num_shape_coefficients_to_fit=-1):
    image_width = im.shape[1]
    image_height = im.shape[0]

    (meshs, poses, shape_coeffs, blendshape_coeffss) = eos.fitting.fit_shape_and_pose(model, blendshapes,
                                                                                   [landmarks], landmark_ids,
                                                                                   landmark_mapper,
                                                                                   [image_width], [image_height],
                                                                                   edge_topology, contour_landmarks,
                                                                                   model_contour,
                                                                                   num_iterations=num_iterations,
                                                                                   num_shape_coefficients_to_fit=num_shape_coefficients_to_fit)
    return meshs[0], poses[0], shape_coeffs, blendshape_coeffss[0]

def getMeshFromMultiLandmarks(landmarkss, ims, num_iterations=5, num_shape_coefficients_to_fit=-1):
    image_widths = []
    image_heights = []
    for im in ims:
        image_widths.append(im.shape[1])
        image_heights.append(im.shape[0])

    return getMeshFromMultiLandmarks_IWH(landmarkss, image_widths, image_heights, num_iterations=num_iterations, num_shape_coefficients_to_fit=num_shape_coefficients_to_fit)

def getMeshFromMultiLandmarks_IWH(landmarkss, image_widths, image_heights, num_iterations=5, num_shape_coefficients_to_fit=-1):
    (meshs, poses, shape_coeffs, blendshape_coeffss) = eos.fitting.fit_shape_and_pose(model, blendshapes,
                                                                                   landmarkss, landmark_ids,
                                                                                   landmark_mapper,
                                                                                   image_widths, image_heights,
                                                                                   edge_topology, contour_landmarks,
                                                                                   model_contour,
                                                                                   num_iterations=num_iterations,
                                                                                   num_shape_coefficients_to_fit=num_shape_coefficients_to_fit)
    return meshs, poses, shape_coeffs, blendshape_coeffss


def getFaceFeatures3D(ims, landmarkss=None, num_iterations=5, num_shape_coefficients_to_fit=-1):
    imswlandmarks = []
    if landmarkss is None or len(ims) != len(landmarkss):
        landmarkss = []
        for im in ims:
            landmarks = getLandmarks(im)
            if landmarks is not None:
                landmarkss.append(landmarks)
                imswlandmarks.append(ims)
    else:
        imswlandmarks = ims

    if len(landmarkss) == 0:
        return None

    meshs, poses, shape_coeffs, blendshape_coeffs = getMeshFromMultiLandmarks(landmarkss, imswlandmarks, num_iterations=num_iterations, num_shape_coefficients_to_fit=num_shape_coefficients_to_fit)

    return shape_coeffs

def getFaceFeatures3D2D(im, landmarks, num_iterations=5, num_shape_coefficients_to_fit=-1):
    mesh, pose, shape_coeffs, blendshape_coeffs = getMeshFromLandmarks(landmarks, im, num_iterations=num_iterations, num_shape_coefficients_to_fit=num_shape_coefficients_to_fit)

    verts = np.array(mesh.vertices)[vertIndices]

    faceFeatures = verts[newFaceLines[:, 0]] - verts[newFaceLines[:, 1]]
    faceFeatures = np.linalg.norm(faceFeatures, axis=1)
    return faceFeatures

def createTextureMap(mesh, pose, im):
    return eos.render.extract_texture(mesh, pose, im)


def exportMeshToJSON(mesh):
    verts = np.array(mesh.vertices)[:,0:3].flatten().tolist()

    uvs = np.array(mesh.texcoords)
    uvs[:,1] = 1-uvs[:,1]
    uvs = uvs.flatten().tolist()

    triangles = np.array(mesh.tvi)
    faces = np.zeros((triangles.shape[0], 1+3+3), dtype=triangles.dtype)
    faces[:, 0] = 8
    faces[:, 1:4] = triangles
    faces[:, 4:7] = triangles
    faces = faces.flatten().tolist()

    outdata = {}
    outdata["metadata"] = {
        "version": 4,
        "type": "geometry",
        "generator": "GeometryExporter"
    }
    outdata["vertices"] = verts
    outdata["uvs"] = [uvs]
    outdata["faces"] = faces

    return outdata

def ensureImageLessThanMax(im, maxsize=512):
    height, width, depth = im.shape
    if height > maxsize or width > maxsize:

        if width > height:
            ratio = maxsize / float(width)
            width = maxsize
            height = int(height * ratio)
        else:
            ratio = maxsize / float(height)
            height = maxsize
            width = int(width * ratio)
        im = cv2.resize(im,(width,height))
    return im