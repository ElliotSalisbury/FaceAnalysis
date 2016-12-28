import eos
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
import cv2
from faceFeatures import getFaceFeatures
import json

def exportToJSON(mesh):
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
    outdata["normals"] = verts
    outdata["uvs"] = [uvs]
    outdata["faces"] = faces

    with open('example.json', 'w') as outfile:
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

def project(p, modelview, proj, viewport):
    tmp = modelview * p[:, np.newaxis]
    tmp = proj * tmp

    tmp = tmp/tmp[3]
    tmp = tmp*0.5 + 0.5
    tmp[0] = tmp[0] * viewport[2] + viewport[0]
    tmp[1] = tmp[1] * viewport[3] + viewport[1]

    return np.array(tmp[0:2]).flatten()

def drawMesh(mesh, pose, isomap, image):
    verts = np.array(mesh.vertices)
    modelview = np.matrix(pose.get_modelview())
    proj = np.matrix(pose.get_projection())
    viewport = np.array([0,image.shape[0], image.shape[1], -image.shape[0]])

    for triangle in mesh.tvi:
        p0, p1, p2 = verts[triangle]

        p02d = project(p0, modelview, proj, viewport).astype(np.int64)
        p12d = project(p1, modelview, proj, viewport).astype(np.int64)
        p22d = project(p2, modelview, proj, viewport).astype(np.int64)

        p02d = (p02d[0], p02d[1])
        p12d = (p12d[0], p12d[1])
        p22d = (p22d[0], p22d[1])

        cv2.line(image, p02d, p12d, (0, 255, 0), thickness=1)
        cv2.line(image, p12d, p22d, (0, 255, 0), thickness=1)
        cv2.line(image, p22d, p02d, (0, 255, 0), thickness=1)

    cv2.imshow("lines", image)

def main():
    # im = cv2.imread("C:\\Users\\Elliot\\Desktop\\fb\\MyFaces\\8.0\\0151.jpg")
    im = cv2.imread("C:\\Users\\Elliot\\Desktop\\test3.jpg")
    im = ensureImageLessThanMax(im, 1024)

    landmarks, faceFeatures = getFaceFeatures(im)

    """Demo for running the eos fitting from Python."""
    # landmarks = read_pts('C:/eos/install/bin/data/image_0010.pts')
    landmark_ids = list(map(str, range(1, 69))) # generates the numbers 1 to 68, as strings
    image_width = im.shape[1] # Make sure to adjust these when using your own images!
    image_height = im.shape[0]

    model = eos.morphablemodel.load_model("C:/eos/install/share/sfm_shape_3448.bin")
    blendshapes = eos.morphablemodel.load_blendshapes("C:/eos/install/share/expression_blendshapes_3448.bin")
    landmark_mapper = eos.core.LandmarkMapper('C:/eos/install/share/ibug2did.txt')
    edge_topology = eos.morphablemodel.load_edge_topology('C:/eos/install/share/sfm_3448_edge_topology.json')
    contour_landmarks = eos.fitting.ContourLandmarks.load('C:/eos/install/share/ibug2did.txt')
    model_contour = eos.fitting.ModelContour.load('C:/eos/install/share/model_contours.json')

    (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(model, blendshapes,
        landmarks, landmark_ids, landmark_mapper,
        image_width, image_height, edge_topology, contour_landmarks, model_contour)

    isomap = eos.render.extract_texture(mesh, pose, im)
    # cv2.imshow("texture", isomap[:,:,0:3])
    cv2.imwrite("example.jpg", isomap)
    exportToJSON(mesh)

    print("done")
    # drawMesh(mesh, pose, isomap, im)
    # cv2.waitKey(-1)

    # Now you can use your favourite plotting/rendering library to display the fitted mesh, using the rendering
    # parameters in the 'pose' variable.

    # Or for example extract the texture map, like this:




def read_pts(filename):
    """A helper function to read ibug .pts landmarks from a file."""
    lines = open(filename).read().splitlines()
    lines = lines[3:71]

    landmarks = []
    for l in lines:
        coords = l.split()
        landmarks.append([float(coords[0]), float(coords[1])])

    return landmarks

if __name__ == "__main__":
    main()
