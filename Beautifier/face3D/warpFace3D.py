import numpy as np
import cv2
from warpFace import warpFace

MESH_LANDMARKS_TO_VERTS=np.array([
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
33,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
255,
229,
233,
2086,
157,
590,
2091,
666,
662,
658,
2842,
379,
272,
114,
100,
2794,
270,
2797,
537,
177,
172,
191,
181,
173,
174,
614,
624,
605,
610,
607,
606,
398,
315,
413,
329,
825,
736,
812,
841,
693,
411,
264,
431,
-1,
416,
423,
828,
-1,
817,
442,
404,
])

MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
ALL_FACE_LANDMARKS = MOUTH_POINTS + RIGHT_BROW_POINTS + LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + NOSE_POINTS
ALL_FACE_MESH_VERTS = MESH_LANDMARKS_TO_VERTS[ALL_FACE_LANDMARKS]
ALL_FACE_MESH_VERTS = np.delete(ALL_FACE_MESH_VERTS, np.where(ALL_FACE_MESH_VERTS == -1)).tolist()

def project(p, modelview, proj, viewport):
    tmp = modelview * p[:, np.newaxis]
    tmp = proj * tmp

    tmp = tmp/tmp[3]
    tmp = tmp*0.5 + 0.5
    tmp[0] = tmp[0] * viewport[2] + viewport[0]
    tmp[1] = tmp[1] * viewport[3] + viewport[1]

    return np.array(tmp[0:2]).flatten()

def projectMeshTo2D(mesh, pose, image):
    verts = np.array(mesh.vertices)
    modelview = np.matrix(pose.get_modelview())
    proj = np.matrix(pose.get_projection())
    viewport = np.array([0,image.shape[0], image.shape[1], -image.shape[0]])

    verts2d = np.zeros((verts.shape[0],2),dtype=np.float64)
    for i, vert in enumerate(verts):
        verts2d[i,:] = project(vert, modelview, proj, viewport)

    return verts2d

def warpFace3D(im, oldMesh, pose, newMesh):
    oldVerts2d = projectMeshTo2D(oldMesh, pose, im)
    oldConvexHullIndexs = cv2.convexHull(oldVerts2d.astype(np.float32), returnPoints=False)

    warpPointIndexs = oldConvexHullIndexs.flatten().tolist() + ALL_FACE_MESH_VERTS
    oldLandmarks = oldVerts2d[warpPointIndexs]

    newVerts2d = projectMeshTo2D(newMesh, pose, im)
    newLandmarks = newVerts2d[warpPointIndexs]

    warpedIm = warpFace(im, oldLandmarks, newLandmarks)

    return warpedIm