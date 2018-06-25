#==== Import the necessary libs ====#
# If libs cannot be found, check the interpreter setting
print("Importing necessary libs...")
from skimage.segmentation import slic # Segments image using K-Means clustering
# in Color-(x,y,z) space
# from skimage.segmentation import mark_boundaries
# from skimage.util import img_as_float
from skimage import io
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np
# Test_Mode = True
use_kmeans = True
if use_kmeans == True:
    use_agglomerativeClustering = False
else:
    use_agglomerativeClustering = True
print("Successfully imported all the libs!")
#========================================================#

#==== Parameter initialization ====#
print("Initializing parameters...")
image_num_range = range(7, 11, 1) # The number of image in the target folder (Start number, End number + 1, Increment step) From 5-9 are test imgs
max_num_segments = 250
ttl_num_segments = 0 # Initialize the ttl number of segments variable
largest_dim_sgmnt_fraction = int(np.load("D:/SLIC/segment_index/largest_dim.npy"))
image = io.imread("C:/utils/pics/image0000.jpg")
image_channel = np.shape(image)[2]
del image
for image_num in image_num_range:
    act_num_segments = np.load("D:/SLIC/segment_index/img%d_act_num_segments.npy" % image_num)
    ttl_num_segments = act_num_segments + ttl_num_segments
sgmnt_fraction_padded = np.zeros((largest_dim_sgmnt_fraction * largest_dim_sgmnt_fraction * image_channel,
                                      ttl_num_segments))
all_sgmnt_ind = 0
#==== K-Mean Clustering ====#
# Read segment fractions iteratively
for image_num in image_num_range:
    print("Loading the segmenation result of image number: %d/%d" %(image_num, np.amax(image_num_range)))
    # Load the actual fractions of segments in the current image
    act_num_segments = np.load("D:/SLIC/segment_index/img%d_act_num_segments.npy" % image_num) # Load the actual number of segment fractions in the current image/segment

    # image = io.imread("C:/utils/Python/SLIC_Tutorial/%d.jpg" % image_num)
    # image = img_as_float(image)
    # segments = np.load("C:/utils/Python/SLIC_Tutorial/segments/segmentImg%d_MaxSegment%d.npy" %(image_num, max_num_segments))
    # # Read all the npy files one by one #
    # segment_index_row_ind = np.shape(segments)[0] - 1 # Calculate the dimension of the current segmented image in row direction. Actual index = total - 1(1080 pixels)
    # segment_index_column_ind = np.shape(segments)[1] -  1 # Calculate the dimension of the current segmented image in column direction. Actual index = total - 1(1920 pixels)
    # act_num_segments = np.amax(segments) + 1 # Determine the actual fractions of segments in the current image
    # # Initialize the matrices to store the segment indices in the image coor sys. The initialized matrix equals to the size of the image to cover all the extreme cases.
    # # For example, one fraction of the segment is as large as the original image.

    # #---- Create two one-row arrays to store the indices ----#
    # sgmnt_fraction_row_index = np.zeros((((segment_index_row_ind+1)*(segment_index_column_ind+1)+1), act_num_segments)) #  Initialize the segment_fraction_row_index matrix. 1080 * 1920 + Act_num_segments (for 9999 storage) per img
    # sgmnt_fraction_column_index = np.zeros((((segment_index_row_ind+1)*(segment_index_column_ind+1)+1), act_num_segments)) # Initialize the segment_fraction_column_index matrix. 1080 * 1920 * Act_num_segments (for 9999 storage) per img
    # #---- Read all the segment npy files one by one END ----#

    for current_segment_num in range(0, act_num_segments, 1):
        print("Reading padded segment fraction: %d/%d of image %d/%d" % (current_segment_num, act_num_segments-1, image_num, np.amax(image_num_range)))
        sgmnt_fraction_padded_buffer = np.zeros((largest_dim_sgmnt_fraction, largest_dim_sgmnt_fraction, image_channel,
                                                 1)) # Create a buffer to store the padded segment fraction read from the
        # Read one padded fraction
        sgmnt_fraction_padded_buffer[:, :, :, 0] = np.load("D:/SLIC/segment_fractions/Img_%d_sgmntPad_%d.npy" %(image_num, current_segment_num))
        sgmnt_fraction_padded_buffer = np.reshape(sgmnt_fraction_padded_buffer, (largest_dim_sgmnt_fraction * largest_dim_sgmnt_fraction * image_channel))
        sgmnt_fraction_padded[:, all_sgmnt_ind] = sgmnt_fraction_padded_buffer
        all_sgmnt_ind = all_sgmnt_ind + 1
del sgmnt_fraction_padded_buffer
sgmnt_fraction_padded = np.transpose(sgmnt_fraction_padded)
print("Clustering...")
if use_kmeans == True:
    kmeans = KMeans(n_clusters=12)
    kmeans.fit(sgmnt_fraction_padded)
elif use_agglomerativeClustering == True:
    agglomerativeClustering = AgglomerativeClustering(n_clusters=12)
    agglomerativeClustering.fit(sgmnt_fraction_padded)
print("Saving clustering results...")
for all_sgmnt_ind in range(0, ttl_num_segments, 1):
    print("Saving clustering result segment: %d/%d" % (all_sgmnt_ind, ttl_num_segments))
    sgmnt_fraction_padded_buffer = np.reshape(sgmnt_fraction_padded[all_sgmnt_ind, :], (largest_dim_sgmnt_fraction, largest_dim_sgmnt_fraction, image_channel))
    # print("Current cluster membership:\n{}".format(kmeans.labels_[all_sgmnt_ind]))
    if use_kmeans == True:
        fig = plt.figure("Cluster membership %d" % kmeans.labels_[all_sgmnt_ind])
    else:
        fig = plt.figure("Cluster membership %d" % agglomerativeClustering.labels_[all_sgmnt_ind])
    plt.imshow(sgmnt_fraction_padded_buffer)
    plt.axis("off")
    if use_kmeans == True:
        fig.savefig(
            "D:/SLIC/Clustering_Results/Membership_%d/Segment fraction %d - Cluster membership %d" % (kmeans.labels_[all_sgmnt_ind], all_sgmnt_ind, kmeans.labels_[all_sgmnt_ind]))
    else:
        fig.savefig(
            "D:/SLIC/Clustering_Results/Membership_%d/Segment fraction %d - Cluster membership %d" % (
                agglomerativeClustering.labels_[all_sgmnt_ind], all_sgmnt_ind, agglomerativeClustering.labels_[all_sgmnt_ind]))