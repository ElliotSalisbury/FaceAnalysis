import numpy as np
import cv2
import sys

FACE_TRIANGLES = np.load("triangles.npy")
CORNER_TRIANGLES = np.load("cornerTriangles.npy")

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri) :
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image

    dst = cv2.warpAffine(src, warpMat, (src.shape[1], src.shape[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(imSrc, imDst, tSrc, tDst) :
    # Get mask by filling triangle
    mask = np.zeros((imDst.shape), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tDst), (1.0, 1.0, 1.0), 16, 0)

    warpImage = applyAffineTransform(imSrc, tSrc, tDst)

    # Copy triangular region of the rectangular patch to the output image
    imDst = imDst*(1-mask) + warpImage*mask
    return imDst

def getCornerTrianglePts(im, landmarks):
    cornerPts = np.array([(0,0), (im.shape[1], 0), (im.shape[1], im.shape[0]), (0, im.shape[0])])

    #convert negative indexs to corner pts
    cornerTriangles2 = np.zeros((CORNER_TRIANGLES.shape[0],3,2))
    for i, tri in enumerate(CORNER_TRIANGLES):
        triPts = np.zeros((3,2))
        for j, t in enumerate(tri):
            if t<0:
                triPts[j,:] = cornerPts[-t - 1]
            else:
                triPts[j, :] = landmarks[t]
        cornerTriangles2[i] = triPts
    return cornerTriangles2

def warpFace(im, oldLandmarks, newLandmarks):
    print("morphing face")
    newIm = im.copy()
    # newIm = np.zeros(im.shape)
    oldCornerTrianglePts = getCornerTrianglePts(im, oldLandmarks)
    newCornerTrianglePts = getCornerTrianglePts(im, newLandmarks)

    for ti in range(len(newCornerTrianglePts)):
        oldTriangle = oldCornerTrianglePts[ti]
        newTriangle = newCornerTrianglePts[ti]

        newIm = warpTriangle(im, newIm, oldTriangle, newTriangle)

    for triangle in FACE_TRIANGLES:
        oldTriangle = oldLandmarks[triangle]
        newTriangle = newLandmarks[triangle]

        newIm = warpTriangle(im, newIm, oldTriangle, newTriangle)
    newIm = np.uint8(newIm)
    return newIm