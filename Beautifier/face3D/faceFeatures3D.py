import eos
import numpy as np
import os
import cv2
from Beautifier.faceFeatures import getLandmarks
import json
from Beautifier.face3D.warpFace3D import warpFace3D, project, renderFaceTo2D
import math

EOS_SHARE_PATH = os.environ['EOS_DATA_PATH']

landmark_ids = list(map(str, range(1, 69)))  # generates the numbers 1 to 68, as strings
model = eos.morphablemodel.load_model(os.path.join(EOS_SHARE_PATH,"sfm_shape_3448.bin"))
blendshapes = eos.morphablemodel.load_blendshapes(os.path.join(EOS_SHARE_PATH,"expression_blendshapes_3448.bin"))
landmark_mapper = eos.core.LandmarkMapper(os.path.join(EOS_SHARE_PATH,"ibug2did.txt"))
edge_topology = eos.morphablemodel.load_edge_topology(os.path.join(EOS_SHARE_PATH,"sfm_3448_edge_topology.json"))
contour_landmarks = eos.fitting.ContourLandmarks.load(os.path.join(EOS_SHARE_PATH,"ibug2did.txt"))
model_contour = eos.fitting.ModelContour.load(os.path.join(EOS_SHARE_PATH,"model_contours.json"))

def getMeshFromLandmarks(landmarks, im):
    image_width = im.shape[1]
    image_height = im.shape[0]

    (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(model, blendshapes,
                                                                                   landmarks, landmark_ids,
                                                                                   landmark_mapper,
                                                                                   image_width, image_height,
                                                                                   edge_topology, contour_landmarks,
                                                                                   model_contour)#, num_shape_coefficients_to_fit=10)
    return mesh, pose, shape_coeffs, blendshape_coeffs

def getFaceFeatures3D(im, landmarks=None):
    if landmarks is None:
        landmarks = getLandmarks(im)
        if landmarks is None:
            return None

    mesh, pose, shape_coeffs, blendshape_coeffs = getMeshFromLandmarks(landmarks, im)

    return shape_coeffs + blendshape_coeffs

def createTextureMap(mesh, pose, im):
    return eos.render.extract_texture(mesh, pose, im)


def exportMeshToJSON(mesh, outpath):
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

    with open(outpath, 'w') as outfile:
        json.dump(outdata, outfile, indent=4, sort_keys=True)

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

def main():
    # im = cv2.imread("C:\\Users\\Elliot\\Desktop\\fb\\MyFaces\\8.0\\0151.jpg")
    im = cv2.imread("C:\\Users\\ellio\\Desktop\\test.png")[:,:,:3]
    im = ensureImageLessThanMax(im, 1024)

    landmarks = getLandmarks(im)
    if landmarks is None:
        return None

    mesh, pose, shape_coeffs, blendshape_coeffs = getMeshFromLandmarks(landmarks, im)
    isomap = createTextureMap(mesh, pose, im)
    cv2.imwrite("./webview/example.jpg", isomap)
    # renderFaceTo2D(im,mesh,pose,isomap)

    newim = np.zeros((im.shape[0] * 5, im.shape[1] * 5, im.shape[2]), np.uint8)
    count = 0
    for i in np.linspace(-3, 3, 5):
        for j in np.linspace(-3, 3, 5):
            new_coeffs = list(shape_coeffs)
            new_coeffs[2] += i
            new_coeffs[3] += j

            newMesh = eos.morphablemodel.draw_sample(model, blendshapes, new_coeffs, blendshape_coeffs, [])

            warpedIm = warpFace3D(im, mesh, pose, newMesh)
            cv2.imshow("warped", warpedIm)
            cv2.imwrite("./webview/example_%i.jpg"%count, warpedIm)
            cv2.waitKey(1)

            x = count % 5
            y = int(count / 5)
            newim[im.shape[0]*y:im.shape[0]*(y+1),im.shape[1]*x:im.shape[1]*(x+1),:]=warpedIm


            exportMeshToJSON(newMesh, "./webview/example_%i.json"%count)
            count += 1
    cv2.imwrite("all.jpg", newim)

if __name__ == "__main__":
    main()
