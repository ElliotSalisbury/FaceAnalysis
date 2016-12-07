import csv
import os
import cv2
import dlib
import numpy as np

demographicscsv = "E:\\Facedata\\10k US Adult Faces Database\\Full Attribute Scores\\demographic & others labels\\demographic-others-labels-final.csv"
imfolder = "E:\\Facedata\\10k US Adult Faces Database\\Face Images"

print("reading data")
demographicsData = []
with open(demographicscsv, 'r') as demoF:
    reader = csv.reader(demoF)
    header = []
    for i, row in enumerate(reader):
        if i == 0:
            header = row
            continue

        row[0] = os.path.join(imfolder, row[0])
        rowDict = {header[j]:row[j] for j in range(len(row)) if os.path.isfile(row[0])}
        demographicsData.append(rowDict)

FACESWAP_SHAPEPREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(FACESWAP_SHAPEPREDICTOR_PATH)

for i, data in enumerate(demographicsData):
    im = cv2.imread(data['Filename'])

    rects = detector(im, 1)
    if len(rects) == 0:
        continue

    landmarks = np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
    landmarks = [(int(p[0, 0]), int(p[0, 1])) for p in landmarks]

    # delaunay triangulate face
    subdiv = cv2.Subdiv2D((0, 0, im.shape[1], im.shape[0]))
    for p in landmarks:
        subdiv.insert(p)

    # Draw delaunay triangles
    triangleList = subdiv.getTriangleList()
    size = im.shape
    r = (0, 0, size[1], size[0])

    # Check if a point is inside a rectangle
    def rect_contains(rect, point):
        if point[0] < rect[0]:
            return False
        elif point[1] < rect[1]:
            return False
        elif point[0] > rect[2]:
            return False
        elif point[1] > rect[3]:
            return False
        return True

    def pteq(p1, p2):
        return p1[0] == p2[0] and p1[1] == p2[1]
    def getptindex(p1):
        for i, p2 in enumerate(landmarks):
            if pteq(p1, p2):
                return i
        return None


    lines = set()
    triangles = set()
    delaunay_color = (255, 0, 255)
    for t in triangleList:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            p1i = getptindex(pt1)
            p2i = getptindex(pt2)
            p3i = getptindex(pt3)

            line1 = (p1i, p2i)
            line2 = (p2i, p3i)
            line3 = (p3i, p1i)

            # dont add backwards lines
            if (line1[1], line1[0]) not in lines:
                lines.add(line1)
            if (line2[1], line2[0]) not in lines:
                lines.add(line2)
            if (line3[1], line3[0]) not in lines:
                lines.add(line3)
            triangles.add((p1i,p2i,p3i))

            cv2.line(im, pt1, pt2, delaunay_color, 1, 0)
            cv2.line(im, pt2, pt3, delaunay_color, 1, 0)
            cv2.line(im, pt3, pt1, delaunay_color, 1, 0)

    nplines = np.zeros((len(lines), 2), dtype=np.uint32)
    for i, line in enumerate(lines):
        pt1 = landmarks[line[0]]
        pt2 = landmarks[line[1]]
        nplines[i, :] = line
        cv2.line(im, pt1, pt2, (0, 0, 255), 1, 0)
    nptriangles = np.zeros((len(triangles), 3), dtype=np.uint32)
    for i, triangle in enumerate(triangles):
        nptriangles[i, :] = triangle

    cv2.imshow("triangles", im)
    cv2.waitKey(-1)

    np.save("lines.npy", nplines)
    np.save("triangles.npy", nptriangles)