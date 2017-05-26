import numpy as np
import msgpack
import os
import scipy.io

def exportModelToJSON(model):
    outdata = {}
    outdata["shapeEV"] = np.array(model.shapeEV).tolist()
    outdata["shapePC"] = np.array(model.shapePC).tolist()
    outdata["shapeMU"] = np.array(model.shapeMU).tolist()
    outdata["faces"] = (model.faces - 1).tolist()

    return outdata

BFM_path = os.path.join(os.environ['CNN_PATH'], 'BaselFaceModel_mod.mat')
model = scipy.io.loadmat(BFM_path, squeeze_me=True, struct_as_record=False)
model = model["BFM"]
outpath = "."
with open(os.path.join(outpath, "bfm.msg"), 'wb') as outfile:
    meshjson = exportModelToJSON(model)
    msgpack.dump(meshjson, outfile)