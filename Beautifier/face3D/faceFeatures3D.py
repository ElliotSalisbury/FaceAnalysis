import eos
import numpy as np
import os
import cv2
from Beautifier.faceFeatures import getLandmarks, faceLines

EOS_SHARE_PATH = os.environ['EOS_DATA_PATH']

LANDMARK_IDS = list(map(str, range(1, 69)))  # generates the numbers 1 to 68, as strings

class FaceFitting:
    def __init__(self, model_path, blendshapes_path, landmarks_mapping_path, edge_topology_path, contour_landmarks_path, model_contour_path):
        self.model = eos.morphablemodel.load_model(model_path)

        if blendshapes_path:
            self.blendshapes = eos.morphablemodel.load_blendshapes(blendshapes_path)
        else:
            self.blendshapes = []

        self.landmark_mapper = eos.core.LandmarkMapper(landmarks_mapping_path)

        self.edge_topology = eos.morphablemodel.load_edge_topology(edge_topology_path)
        self.contour_landmarks = eos.fitting.ContourLandmarks.load(contour_landmarks_path)
        self.model_contour = eos.fitting.ModelContour.load(model_contour_path)

        self.landmarks_2_vert_indices = [self.landmark_mapper.convert(l) for l in LANDMARK_IDS]
        self.landmarks_2_vert_indices = np.array([int(i) if i else -1 for i in self.landmarks_2_vert_indices])

    def getMeshFromLandmarks(self, landmarks, im, num_iterations=50, num_shape_coefficients_to_fit=-1, shape_coeffs_guess=[], blendshape_coeffs_guess=[]):
        image_width = im.shape[1]
        image_height = im.shape[0]

        if blendshape_coeffs_guess:
            blendshape_coeffs_guess = [blendshape_coeffs_guess]

        (meshs, poses, shape_coeffs, blendshape_coeffss) = self.getMeshFromMultiLandmarks_IWH([landmarks], [image_width], [image_height],
                                                                                       num_iterations=num_iterations,
                                                                                       num_shape_coefficients_to_fit=num_shape_coefficients_to_fit,
                                                                                       shape_coeffs_guess=shape_coeffs_guess,
                                                                                       blendshape_coeffs_guess=blendshape_coeffs_guess)
        return meshs[0], poses[0], shape_coeffs, blendshape_coeffss[0]

    def getMeshFromMultiLandmarks(self, landmarkss, ims, num_iterations=5, num_shape_coefficients_to_fit=-1, shape_coeffs_guess=[], blendshape_coeffs_guess=[]):
        image_widths = []
        image_heights = []
        for im in ims:
            image_widths.append(im.shape[1])
            image_heights.append(im.shape[0])

        return self.getMeshFromMultiLandmarks_IWH(landmarkss, image_widths, image_heights, num_iterations=num_iterations, num_shape_coefficients_to_fit=num_shape_coefficients_to_fit, shape_coeffs_guess=shape_coeffs_guess, blendshape_coeffs_guess=blendshape_coeffs_guess)

    def getMeshFromMultiLandmarks_IWH(self, landmarkss, image_widths, image_heights, num_iterations=5, num_shape_coefficients_to_fit=-1, shape_coeffs_guess=[], blendshape_coeffs_guess=[]):
        (meshs, poses, shape_coeffs, blendshape_coeffss) = eos.fitting.fit_shape_and_pose(self.model, self.blendshapes,
                                                                                          landmarkss, LANDMARK_IDS,
                                                                                          self.landmark_mapper,
                                                                                          image_widths, image_heights,
                                                                                          self.edge_topology, self.contour_landmarks,
                                                                                          self.model_contour,
                                                                                          num_iterations=num_iterations,
                                                                                          num_shape_coefficients_to_fit=num_shape_coefficients_to_fit,
                                                                                          pca_shape_coefficients=shape_coeffs_guess,
                                                                                          blendshape_coefficients = blendshape_coeffs_guess)
        return meshs, poses, shape_coeffs, blendshape_coeffss


    def getFaceFeatures3D(self, ims, landmarkss=None, num_iterations=5, num_shape_coefficients_to_fit=-1):
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

        meshs, poses, shape_coeffs, blendshape_coeffs = self.getMeshFromMultiLandmarks(landmarkss, imswlandmarks, num_iterations=num_iterations, num_shape_coefficients_to_fit=num_shape_coefficients_to_fit)

        return shape_coeffs

    def getMeshFromShapeCeoffs(self, shape_coeffs=[], blendshape_coeffs=[]):
        return eos.morphablemodel.draw_sample(self.model, self.blendshapes, shape_coeffs, blendshape_coeffs, [])
    def getPoseFromShapeCeoffs(self, landmarks, im, shape_coeffs, blendshape_coeffs=[]):
        image_width = im.shape[1]
        image_height = im.shape[0]

        return eos.fitting.fit_pose(self.model, landmarks, LANDMARK_IDS, self.landmark_mapper, image_width, image_height, shape_coeffs, blendshapes=self.blendshapes, blendshape_coefficients=blendshape_coeffs)


# model = eos.morphablemodel.load_model()
# blendshapes = eos.morphablemodel.load_blendshapes()
# landmark_mapper = eos.core.LandmarkMapper()
# edge_topology = eos.morphablemodel.load_edge_topology()
# contour_landmarks = eos.fitting.ContourLandmarks.load()
# model_contour = eos.fitting.ModelContour.load()
# model_bfm = eos.morphablemodel.load_model(os.path.join(EOS_SHARE_PATH,"bfm_small.bin"))
# landmark_mapper_bfm = eos.core.LandmarkMapper(os.path.join(EOS_SHARE_PATH,"ibug_to_bfm_small.txt"))
# landmarks_2_vert_indices_bfm = [landmark_mapper_bfm.convert(l) for l in landmark_ids]
# landmarks_2_vert_indices_bfm = np.array([int(i) if i else -1 for i in landmarks_2_vert_indices_bfm])
SFM_FACEFITTING = FaceFitting(model_path=os.path.join(EOS_SHARE_PATH, "sfm_shape_3448.bin"),
                              blendshapes_path=os.path.join(EOS_SHARE_PATH, "expression_blendshapes_3448.bin"),
                              landmarks_mapping_path=os.path.join(EOS_SHARE_PATH, "ibug_to_sfm.txt"),
                              edge_topology_path=os.path.join(EOS_SHARE_PATH, "sfm_3448_edge_topology.json"),
                              contour_landmarks_path=os.path.join(EOS_SHARE_PATH, "ibug_to_sfm.txt"),
                              model_contour_path=os.path.join(EOS_SHARE_PATH,"model_contours.json"))

BFM_FACEFITTING = FaceFitting(model_path=os.path.join(EOS_SHARE_PATH, "bfm_small.bin"),
                              blendshapes_path=None,
                              landmarks_mapping_path=os.path.join(EOS_SHARE_PATH, "ibug_to_bfm_small.txt"),
                              edge_topology_path=os.path.join(EOS_SHARE_PATH, "sfm_3448_edge_topology.json"),
                              contour_landmarks_path=os.path.join(EOS_SHARE_PATH, "ibug_to_bfm_small.txt"),
                              model_contour_path=os.path.join(EOS_SHARE_PATH, "model_contours.json"))

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