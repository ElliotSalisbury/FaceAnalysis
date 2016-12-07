import numpy as np
import cv2
import sys

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

if __name__ == '__main__' :

    filename1 = 'hillary_clinton.jpg'
    filename2 = 'ted_cruz.jpg'
    alpha = 0.5

    # Read images
    img1 = cv2.imread(filename1);
    img2 = cv2.imread(filename2);

    # Convert Mat to float data type
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    # Read array of corresponding points
    points1 = readPoints(filename1 + '.txt')
    points2 = readPoints(filename2 + '.txt')
    points = [];

    # Compute weighted average point coordinates
    for i in xrange(0, len(points1)):
        x = ( 1 - alpha ) * points1[i][0] + alpha * points2[i][0]
        y = ( 1 - alpha ) * points1[i][1] + alpha * points2[i][1]
        points.append((x,y))


    # Allocate space for final output
    imgMorph = np.zeros(img1.shape, dtype = img1.dtype)

    # Read triangles from tri.txt
    with open("tri.txt") as file :
        for line in file :
            x,y,z = line.split()

            x = int(x)
            y = int(y)
            z = int(z)

            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [ points[x], points[y], points[z] ]

            # Morph one triangle at a time.
            morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)


    # Display Result
    cv2.imshow("Morphed Face", np.uint8(imgMorph))
    cv2.waitKey(0)