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
    model = model.get_shape_model()

    outdata = {}
    outdata["shapeEV"] = np.array(model.get_eigenvalues()).tolist()
    outdata["shapePC"] = np.array(model.get_rescaled_pca_basis()).tolist()
    outdata["shapeMU"] = np.array(model.get_mean()).tolist()
    outdata["faces"] = np.array(model.get_triangle_list()).tolist()

    return outdata

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