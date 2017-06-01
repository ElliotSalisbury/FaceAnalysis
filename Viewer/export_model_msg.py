import numpy as np
import msgpack
import os
import scipy.io
from Beautifier.face3D.faceFeatures3D import model as model_sfm

def exportModelBFMToJSON(model):
    outdata = {}
    outdata["shapeEV"] = np.array(model.shapeEV).tolist()
    outdata["shapePC"] = np.array(model.shapePC).tolist()
    outdata["shapeMU"] = np.array(model.shapeMU).tolist()
    outdata["faces"] = (model.faces - 1).tolist()

    return outdata

def exportModelSFMToJSON(model):
    shape_pca_model = model.get_shape_model()

    outdata = {}
    outdata["shapeEV"] = np.array(shape_pca_model.get_eigenvalues()).tolist()
    outdata["shapePC"] = np.array(shape_pca_model.get_rescaled_pca_basis()).tolist()
    outdata["shapeMU"] = np.array(shape_pca_model.get_mean()).tolist()
    outdata["faces"] = np.array(shape_pca_model.get_triangle_list()).tolist()
    outdata["UVs"] = np.array(model.get_texture_coordinates()).tolist()

    return outdata

def getSmallerBFM(S, model):
    landmarksToConvert = [43783,38057,38530,38775,39320,39581,39949,8319,6653,8334,9878,2088,5959,10603,14472,5006,8344,11714,8374,8354,8366]
    smallestLandmarkDistance = np.ones(len(landmarksToConvert)) * 10000
    smallestLandmarkIndex = np.ones(len(landmarksToConvert)) * -1

    with open("./bfm_mean_small.obj", "r") as objfile:
        verts = []
        faces = []
        for line in objfile.readlines():
            if line[0] == "#":
                continue
            id, *rest = line.split(" ")

            if id == "v":
                vert = [float(c) for c in rest]
                verts.append(vert)

            if id == "f":
                face = [int(c) for c in rest]
                faces.append(face)

    verts = np.array(verts)
    faces = np.array(faces) - 1

    vert_indexs = []
    for newI, vert in enumerate(verts):
        deltas = S - vert
        distances = np.linalg.norm(deltas, axis=1)
        index = np.argmin(distances)
        vert_indexs.append(index)

        landmarkDistances = distances[landmarksToConvert]
        smallerLandmarks = np.where(landmarkDistances < smallestLandmarkDistance)
        smallestLandmarkDistance[smallerLandmarks] = landmarkDistances[smallerLandmarks]
        smallestLandmarkIndex[smallerLandmarks] = newI

    newShapePC = []
    newShapeMU = []
    modelShapePC = np.array(model.shapePC)
    modelShapePC = modelShapePC * np.array(model.shapeEV)
    modelShapePC = modelShapePC.tolist()
    modelShapeMU = np.array(model.shapeMU).tolist()
    for index in vert_indexs:
        i = index * 3
        newShapePC.append(modelShapePC[i])
        newShapePC.append(modelShapePC[i+1])
        newShapePC.append(modelShapePC[i+2])
        newShapeMU.append(modelShapeMU[i])
        newShapeMU.append(modelShapeMU[i+1])
        newShapeMU.append(modelShapeMU[i+2])

    outdata = {}
    outdata["shapeEV"] = np.array(model.shapeEV).tolist()
    outdata["shapePC"] = newShapePC
    outdata["shapeMU"] = newShapeMU
    outdata["faces"] = faces.tolist()

    return outdata

def toCerealBinary(outdata, outpath):
    import struct
    with open(outpath, 'wb') as outfile:
        #output the sizes
        outfile.write(struct.pack('i', len(outdata['shapeMU'])))
        outfile.write(struct.pack('i', len(outdata['shapePC'][0])))

        #mean
        outfile.write(struct.pack('{}f'.format(len(outdata['shapeMU'])), *outdata['shapeMU']))

        #convert rescaled pca basis to orthonormal
        PCA = np.array(outdata['shapePC'])
        PCA = PCA * 1/np.sqrt(np.array(outdata['shapeEV']))
        PCA = PCA.T
        for basis in PCA:
            outfile.write(struct.pack('{}f'.format(len(basis)), *basis))

        #eigen values
        outfile.write(struct.pack('{}f'.format(len(outdata['shapeEV'])), *outdata['shapeEV']))

        #num triangles and triangle list
        outfile.write(struct.pack('i', len(outdata['faces'])))
        faces = np.array(outdata['faces']) + 1
        for face in faces:
            outfile.write(struct.pack('{}i'.format(len(face)), *face))

        #do the same for the color model
        # output the sizes
        outfile.write(struct.pack('i', len(outdata['shapeMU'])))
        outfile.write(struct.pack('i', len(outdata['shapePC'][0])))

        # mean
        outfile.write(struct.pack('{}f'.format(len(outdata['shapeMU'])), *outdata['shapeMU']))

        # pca basis
        PCA = np.array(outdata['shapePC']).T
        for basis in PCA:
            outfile.write(struct.pack('{}f'.format(len(basis)), *basis))

        # eigen values
        outfile.write(struct.pack('{}f'.format(len(outdata['shapeEV'])), *outdata['shapeEV']))


BFM_path = os.path.join(os.environ['CNN_PATH'], 'BaselFaceModel_mod.mat')
model_bfm = scipy.io.loadmat(BFM_path, squeeze_me=True, struct_as_record=False)
model_bfm = model_bfm["BFM"]
outpath = "."
with open(os.path.join(outpath, "bfm.msg"), 'wb') as outfile:
    meshjson = exportModelBFMToJSON(model_bfm)
    msgpack.dump(meshjson, outfile)

outpath = "."
with open(os.path.join(outpath, "sfm.msg"), 'wb') as outfile:
    meshjson = exportModelSFMToJSON(model_sfm)
    msgpack.dump(meshjson, outfile)

import Beautifier.faceCNN.utils as utils
S, T = utils.projectBackBFM(model_bfm, np.zeros(198))
outfile = os.path.join(outpath, "bfm_mean.ply")
utils.write_ply(outfile, S, T, (model_bfm.faces - 1))

with open(os.path.join(outpath, "bfm_small.msg"), 'wb') as outfile:
    meshjson = getSmallerBFM(S, model_bfm)
    msgpack.dump(meshjson, outfile)

    toCerealBinary(meshjson, "bfm_small.raw")
