import os
import numpy as np
import caffe
import cv2
import Beautifier.faceCNN.utils as utils
from Beautifier.faceFeatures import getLandmarks

os.environ['GLOG_minloglevel'] = '2'

# CNN network spec
deploy_path = os.path.join(os.environ['CNN_PATH'], 'CNN/deploy_network.prototxt')
model_path  = os.path.join(os.environ['CNN_PATH'], 'CNN/3dmm_cnn_resnet_101.caffemodel')
mean_path = os.path.join(os.environ['CNN_PATH'], 'CNN/mean.binaryproto')
layer_name      = 'fc_ftnew'
## Modifed Basel Face Model
BFM_path = os.path.join(os.environ['CNN_PATH'], 'BaselFaceModel_mod.mat')
## CNN template size
trg_size = 224

caffe.set_mode_cpu()
## Opening mean average image
proto_data = open(mean_path, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean = caffe.io.blobproto_to_array(a)[0]
## Loading the CNN
net = caffe.Classifier(deploy_path, model_path)
## Setting up the right transformer for an input image
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)
transformer.set_mean('data',mean)
print('> CNN Model loaded to regress 3D Shape and Texture!')

def preprocessIm(im, landmarks):
    lms_x = landmarks[:,0]
    lms_y = landmarks[:,1]
    cropped = utils.cropImg(im, min(lms_x), min(lms_y), max(lms_x), max(lms_y))

    resized = cv2.resize(cropped, (trg_size, trg_size))

    return resized.astype(np.float32) / 255

def feedImToNet(im):
    ## Transforming the image into the right format
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    ## Forward pass into the CNN
    net_output = net.forward()
    ## Getting the output
    features = np.hstack([net.blobs[layer_name].data[0].flatten()])

    return features

def getFaceFeaturesCNN(ims, landmarkss=None):
    imswlandmarks = []
    if landmarkss is None or len(ims) != len(landmarkss):
        landmarkss = []
        for im in ims:
            landmarks = getLandmarks(im)
            if landmarks is not None:
                landmarkss.append(landmarks)
                imswlandmarks.append(im)
    else:
        imswlandmarks = ims

    if len(landmarkss) == 0:
        return None

    allFeatures = []
    for i, im in enumerate(imswlandmarks):
        landmarks = landmarkss[i]

        #preprocess im for network
        netim = preprocessIm(im, landmarks)

        #use network to get features
        features = feedImToNet(netim)
        allFeatures.append(features)

    allFeatures = np.array(allFeatures).mean(axis=0)

    return allFeatures

if __name__ == "__main__":
    import glob
    import scipy.io

    # ## Loading the Basel Face Model to write the 3D output
    model = scipy.io.loadmat(BFM_path,squeeze_me=True,struct_as_record=False)
    model = model["BFM"]
    faces = model.faces-1

    dstFolder = "./results/"
    for i, impath in enumerate(glob.glob(r"E:\Facedata\10k US Adult Faces Database\Publication Friendly 49-Face Database\49 Face Images\*.jpg")):
        print(i)
        im = cv2.imread(impath)
        im2 = caffe.io.load_image(impath)
        filename = os.path.basename(impath)

        features = getFaceFeaturesCNN([im])

        outfile = os.path.join(dstFolder, "%s.ply" % filename)

        S, T = utils.projectBackBFM(model, features)
        utils.write_ply(outfile, S, T, faces)