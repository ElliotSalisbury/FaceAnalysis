import numpy as np
import cv2
import os
import scipy

scriptFolder = os.path.dirname(os.path.realpath(__file__))

def findBestFeaturesOptimisation(featuresCNN, gp):
    print("finding optimal face features optimisation")

    def GPCostFunction(features):
        y_pred = gp.predict([features])
        return -y_pred

    bounds = np.zeros((featuresCNN.shape[0],2))
    bounds[:, 0] = -200#featuresCNN - 0.5
    bounds[:, 1] = 200#featuresCNN + 0.5

    optimalNewFaceFeatures = scipy.optimize.minimize(GPCostFunction, featuresCNN, bounds=bounds, method='SLSQP', options={"maxiter": 5, "eps": 0.001})
    return optimalNewFaceFeatures.x

if __name__ == "__main__":
    import glob
    from US10k.US10k import loadUS10k
    from RateMe.RateMe import loadRateMe
    import Beautifier.faceCNN.utils as utils
    import scipy.io
    from Beautifier.faceCNN.faceFeaturesCNN import getFaceFeaturesCNN

    # ## Loading the Basel Face Model to write the 3D output
    BFM_path = os.path.join(os.environ['CNN_PATH'], 'BaselFaceModel_mod.mat')
    attr_path = os.path.join(os.environ['CNN_PATH'], '04_attributes.mat')
    model = scipy.io.loadmat(BFM_path, squeeze_me=True, struct_as_record=False)
    attributes = scipy.io.loadmat(attr_path, squeeze_me=True, struct_as_record=False)
    model = model["BFM"]
    faces = model.faces - 1

    GENDER = "F"

    dstFolder = "./resultsCNN{}/".format(GENDER)

    datasets = [#loadUS10k(type="cnn", gender=GENDER),
                loadRateMe(type="cnn", gender=GENDER)]

    print("begin beautification")
    for i, impath in enumerate(glob.glob(r"E:\Facedata\10k US Adult Faces Database\Publication Friendly 49-Face Database\49 Face Images\*.jpg")):
        print(i)
        im = cv2.imread(impath)
        filename = os.path.basename(impath)

        trainX, trainY, pca, gp = datasets[0]

        try:
            facefeatures = getFaceFeaturesCNN([im])
            newfacefeatures = facefeatures.copy()

            scale = 5
            newfacefeatures[0:99] = newfacefeatures[0:99] - (attributes['gender_shape'][0:99] * scale)
            newfacefeatures[0:99] = newfacefeatures[0:99] - (attributes['weight_shape'][0:99] * scale)
            newfacefeatures[0:99] = newfacefeatures[0:99] + (attributes['height_shape'][0:99] * scale)
            newfacefeatures[0:99] = newfacefeatures[0:99] - (attributes['age_shape'][0:99] * scale)

            # newfacefeatures = facefeatures.copy()
            newfacefeatures[0:99] = findBestFeaturesOptimisation(facefeatures[0:99], gp)

            outfileOLD = os.path.join(dstFolder, "%s_OLD.ply" % filename)
            outfileNEW = os.path.join(dstFolder, "%s_NEW.ply" % filename)

            S, T = utils.projectBackBFM(model, facefeatures)
            utils.write_ply(outfileOLD, S, T, faces)

            S, T = utils.projectBackBFM(model, newfacefeatures)
            utils.write_ply(outfileNEW, S, T, faces)
        except Exception as e:
            print(e)