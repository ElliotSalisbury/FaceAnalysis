import numpy as np
import cv2
from Beautifier.warpFace import warpFace, warpTriangle

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

def getVisibleFacesIndexs(mesh, pose):
    verts = np.array(mesh.vertices)[:,:3]
    norms = np.ones((len(mesh.tvi), 4), dtype=np.float64)
    modelview = np.matrix(pose.get_modelview())
    modelview[0, 3] = 0
    modelview[1, 3] = 0

    for i, triangle in enumerate(mesh.tvi):
        p0, p1, p2 = verts[triangle]
        v1 = p1 - p0
        v2 = p2 - p1

        norm = np.cross(v1,v2)
        norm = norm / np.linalg.norm(norm)
        norms[i,:3] = norm

    rotatedNorms = np.zeros_like(norms)
    for i, norm in enumerate(norms):
        rotatedNorms[i] = np.array(modelview * norm[:, np.newaxis]).flatten()

    return np.where(rotatedNorms[:,2] > 0)

def renderFaceTo2D(im, mesh, pose, isomap):
    verts2d = projectMeshTo2D(mesh, pose, im)
    uvcoords = np.array(mesh.texcoords) * np.array([isomap.shape[1], isomap.shape[0]])

    visibleFaceIndexs = getVisibleFacesIndexs(mesh, pose)
    visibleVertIndexs = np.unique(np.array(mesh.tvi)[visibleFaceIndexs].flatten())

    renderedFace = warpFace(isomap[:,:,:3], uvcoords[visibleVertIndexs], verts2d[visibleVertIndexs], justFace=True, output_shape=(im.shape[0], im.shape[1]))
    meshFace = drawMesh(im, mesh, pose, isomap)
    cv2.imshow("orig", im)
    cv2.imshow("mesh", meshFace)


    blackIs = np.where(
        np.logical_and(np.logical_and(renderedFace[:, :, 0] == 0, renderedFace[:, :, 1] == 0), renderedFace[:, :, 2] == 0))
    renderedFace[blackIs] = im[blackIs]
    cv2.imshow("rendered", renderedFace)

    # warpFace3D(im, mesh, pose, newMesh)

    renderedFace2 = np.zeros_like(im)
    for i, triangle in enumerate(np.array(mesh.tvi)[visibleFaceIndexs]):
        # if i > 500:
        #     break
        srcT = uvcoords[triangle].astype(np.int64)
        dstT = verts2d[triangle].astype(np.int64)

        renderedFace2 = warpTriangle(isomap[:,:,:3], renderedFace2, srcT, dstT)
    cv2.imshow("rendered2", renderedFace2.astype(np.uint8))

    cv2.waitKey(-1)

def drawMesh(im, mesh, pose, isomap):
    verts2d = projectMeshTo2D(mesh, pose, im)
    visibleFaceIndexs = getVisibleFacesIndexs(mesh, pose)

    drawIm = im.copy()
    for triangle in np.array(mesh.tvi)[visibleFaceIndexs]:
        p0, p1, p2 = verts2d[triangle].astype(np.int64)

        p02d = (p0[0], p0[1])
        p12d = (p1[0], p1[1])
        p22d = (p2[0], p2[1])

        cv2.line(drawIm, p02d, p12d, (0, 255, 0), thickness=1)
        cv2.line(drawIm, p12d, p22d, (0, 255, 0), thickness=1)
        cv2.line(drawIm, p22d, p02d, (0, 255, 0), thickness=1)

    return drawIm

def warpFace3D(im, oldMesh, pose, newMesh):
    oldVerts2d = projectMeshTo2D(oldMesh, pose, im)
    oldConvexHullIndexs = cv2.convexHull(oldVerts2d.astype(np.float32), returnPoints=False)

    warpPointIndexs = oldConvexHullIndexs.flatten().tolist() + ALL_FACE_MESH_VERTS
    oldLandmarks = oldVerts2d[warpPointIndexs]

    newVerts2d = projectMeshTo2D(newMesh, pose, im)
    newLandmarks = newVerts2d[warpPointIndexs]

    warpedIm = warpFace(im, oldLandmarks, newLandmarks)

    return warpedIm