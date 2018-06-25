# ==== Import libs ====#
print("[INFO] Importing necessary libraries...")
from keras.preprocessing import image as image_utils
from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
from vgg16 import VGG16
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt


manualImageName = False
# ==== Construct the argument parse and parse the arguments ==== #
ap = argparse.ArgumentParser() # Create a new argument parser object. Decriptions etc of the program
if manualImageName == True:
    ap.add_argument("-i", "--image", required=True,
                    help="path to the input image")
else:
    ap.add_argument("-i", "--image", required=False,
                    help="path to the input image") # Input path of the image. This line is for accessing the program from DOS.
                    # required = True, must input the path first (normally in DOS), = False, bypass this line
                    # in DOS type: python imageNet_Trial.py --image TargetImageNameHere, then Enter to feed the image to the program
args = vars(ap.parse_args()) # Assign a string to "args"
# ==== Load the original image ==== #
# Load the image via OpenCV first
if manualImageName == True:
    orig = cv2.imread(args["image"]) # Load an image using OpenCV. Uncomment this line after debugging
else:
    orig = cv2.imread("C:/utils/Python/deeplearningTest/venv/beer.jpg") # Read image directly
    # cv2.imshow('Oringial Image',orig)
    # cv2.waitKey(0)

# Load the input image using the Keras helper utility while ensuring
# that the image is resized to 224 by 224 pixels, as required by the
# input dimensiton of the NN -- then convert the PIL iamge to a
# NumPy array
print("[INFO] Loading and preprocessing image...")
if manualImageName == True:
    image = image_utils.load_img(args["image"], target_size=(224, 224)) # Resize the image
else:
    image = image_utils.load_img("C:/utils/Python/deeplearningTest/venv/beer.jpg", target_size=(224, 224))
image = image_utils.img_to_array(image) # Convert the image to array
# plt.imshow(image)
# plt.show()

# ==== Image preprocessing = (224, 224, 3), we need to expand the dimension to
# (1, 224, 224, 3), which stands for (Number of images in the test batch, Pixel number in X direction, Pixel number in Y direction, RGB three color channels).
#  Also, we will preprocess the iamge by subtracting the mean
# RGB pixel intensity from the ImageNet dataset
image = np.expand_dims(image, axis=0) # Expand the dimension of the image array
# axis: int, Position in the expanded axes where the new axis is placed
# np.expand_dims returns an expanded ndarray in the specified direction.

image = preprocess_input(image) # Subtract the mean of the image
# plt.imshow(image[0,:,:,0]) # Display the R channel of the 1st image
# plt.show() # Show the image window, without this line, the window that shows the image would close immediately, so
             # we cannot see the picture. Close the window to continue the program.

# ==== Load the VGG16 NN and classify the images ==== #
# Load the VGG16 NN
print("[INFO] Loading network...")
model = VGG16(weights="imagenet") # Import the VGG16 model, import the pre-trained weights

# Classify the input image
print("[INFO] Classifying image...")
preds = model.predict(image) # Feed the image to the network and give the prediction
(inID, label) = decode_predictions(preds)[0] # decode_predictions returns 3 classes, and only "read" the first one ([0])

# P = decode_predictions(preds)
# print(P)
# (label, prob) = P[0][0]

# Display the predictions to our screen
print("ImageNet ID: {}, Label: {}.".format(inID, label))
cv2.putText(orig, "Label: {}".format(label), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) # Display text on the image
cv2.imshow("Classification", orig) # Show the classification result of the original image
                                   # the window name is Classification.
cv2.waitKey(0) # Wait the user to tap the keyboard