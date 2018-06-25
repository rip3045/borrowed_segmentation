import numpy as np
import json

from keras.utils.data_utils import get_file
from keras import backend as K

CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'


def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x


def decode_predictions(preds, top=5):
    #---- My code ----#
    My_Code = True # use my code or not
    #-------------------#
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        #---- My code ----#
        if My_Code == True:
            fpath = "C:/utils/Python/deeplearningTest/venv/imagenet_class_index.json" # "imagenet_class_index.json"
        else:
            fpath = get_file('imagenet_class_index.json',
                             CLASS_INDEX_PATH,
                             cache_subdir='models')
        #---------------------------------------------#
        #---- Origianl code ----#
        # fpath = get_file('imagenet_class_index.json',
        #                  CLASS_INDEX_PATH,
        #                  cache_subdir='models')
        #--------------------------------------------#
        #---- My code ----#
        if My_Code == True:
            CLASS_INDEX = json.load(open(fpath))
            print(CLASS_INDEX['782']) # Shows how to use key to access the element in a dictionary
        else:
            CLASS_INDEX = json.load(open(fpath))
        #-------------------------------------------#
        #---- Original code ----#
        # CLASS_INDEX = json.load(open(fpath))
        #-------------------------------------------#
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        #---- My code ----#
        # The reason that I need to write the program in this way is that the "CLASS_INDEX" (Read from the image_class_index.json file)is a Python dictionary,
        # to access the elements in a dictionary in Python, a string should be used instead of using a integer. So, to loop through the picked "top X" prediction
        # results, variable "i" should be converted to string from integer first. Then, the string "i" can be used to index and access the target element in the
        # "CLASS_INDEX" dictionary.
        for i in top_indices:
            result_index = str(i) # Convert i (integer) to a string
            result = CLASS_INDEX[result_index] # Read the dictionary and query the elecment according to the "index" (called key in a dictionary)
            results.append(result) # Write the result to a list that contains all the top X prediction results.
        # Loop through the top X prediction results. The top X prediction results give you more flexibility to see the actual performance
        # of the trained model. In my opinion, top 1 should be enough for a real implementation case, because we only care about the prediction result with
        # the highest probability. However, seeing other prediction results with high probability can be used to evaluate the performance of the network.
        # That's why we have a top_indices here. In a real application, just pick the prediction with the highest probability as the final output result.
        #--------------------------------------------------------------------------------------------------------------#
        #---- Original code ----#
        # for i in top_indices:
        #     result = [tuple(CLASS_INDEX[i]) + (pred[i],)]
        #     results.append(result)
        #--------------------------------------------------#
        return results
