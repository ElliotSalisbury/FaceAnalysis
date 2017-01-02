import numpy as np
import cv2
import sys
from skimage.transform import PiecewiseAffineTransform, warp
import skimage
import os

scriptFolder = os.path.dirname(os.path.realpath(__file__))

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, dst) :
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image

    dst = cv2.warpAffine(src, warpMat, (dst.shape[1], dst.shape[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(imSrc, imDst, tSrc, tDst) :
    # Get mask by filling triangle
    mask = np.zeros((imDst.shape), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tDst), (1.0, 1.0, 1.0), 16, 0)

    warpImage = applyAffineTransform(imSrc, tSrc, tDst, imDst)

    # Copy triangular region of the rectangular patch to the output image
    imDst = imDst*(1-mask) + warpImage*mask
    return imDst

def warpFace(im, oldLandmarks, newLandmarks, justFace=False, output_shape=None):
    print("warping face")
    if not justFace:
        cornerPts = np.array([(0, 0), (im.shape[1], 0), (im.shape[1], im.shape[0]), (0, im.shape[0])])

        oldLandmarks = np.append(oldLandmarks, cornerPts, axis=0)
        newLandmarks = np.append(newLandmarks, cornerPts, axis=0)

    tform = PiecewiseAffineTransform()
    tform.estimate(newLandmarks,oldLandmarks)

    warped = warp(im, tform, output_shape=output_shape)
    warped = skimage.img_as_ubyte(warped)
    return warped