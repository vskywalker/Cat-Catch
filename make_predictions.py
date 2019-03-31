from dnn_app_utils_v3 import predict
import time
import numpy as np
import h5py
import pickle
from matplotlib.pyplot import imread
import scipy
from PIL import Image
from scipy import ndimage

file_name = "models/parameters/L_layer_model_parameters.pkl"
my_image = "cat2.jpg" # change this to the name of your image file 
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)

def load_model(file_name):
    with open(file_name, "rb") as inp:
        parameters = pickle.load(inp)
    return parameters

def make_prediction(my_image, my_label_y, parameters):
    num_px = 64
    fname = "images/" + my_image
    image = np.array(imread(fname))
    my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((num_px*num_px*3,1))
    my_image = my_image/255.
    my_predicted_image = predict(my_image, my_label_y, parameters)
    return my_predicted_image

parameters = load_model(file_name)
prediction = make_prediction(my_image, my_label_y, parameters)

print(prediction)
