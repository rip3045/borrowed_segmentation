#==== Import the necessary libs ====#
# If libs cannot be found, check the interpreter setting
print("Importing necessary libs...")
from skimage.segmentation import slic # Segments image using K-Means clustering
# in Color-(x,y,z) space
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
Test_Mode = False
not_run_super_pixel = True
not_run_extract_sgmnt_index = True
not_run_pad_sgmnt_fractions = False
#========================================================#

# ==== Initialize all the needed parameters ====#
print("Initializing the needed parameters...")
max_iters = 10  # Max iterations in the SLIC process
save_sgmnt_rslt = True  # Saving(True)/Not saving(False) segmentation results
image_num_range = range(0, 251,
                        1)  # The number of image in the target folder (Start number, End number + 1, Increment step) From 5-9 are test imgs
max_num_segments = 250  # Max number of segments in each image. 300 is enough for large parts
#================================================#

#==== Image segmentation ====#
if not_run_super_pixel == False:
    if save_sgmnt_rslt == True:
        print("Save segmentation results: Yes")
    else:
        print("Save segmentation results: No")
    #==== Initialization END ====#
    # Loop over all the images
    print("Start superpixel image segmentation...")
    for image_num in image_num_range:
        print("Current image number: %d/%d" %(image_num, np.amax(image_num_range)))
    # Apply ALIC and extract (approximately) the supplied number of segments
    # Load an image manually
    # ["%04d" % x for x in range(10000)] # Example: Loop over 0000 to 9999. Padding the empty digits with 0s.

        if Test_Mode == True:
            image = io.imread("C:/utils/Python/SLIC/%d.jpg" % image_num) # Mix a changing number with string- use "String..%d" % var
        else:
            image = io.imread("C:/utils/pics/image%04d.jpg" % image_num) # For images in pics folder
        image = img_as_float(image)

        # Apply ALIC and extract (approximately) the supplied number
        # of segments
        segments = slic(image, n_segments = max_num_segments, max_iter= max_iters, sigma = 5)
        # image: 2D, 3D or 4D nd array
        # n_segments; The approximate number of labels in the segmented
        # output image. Or say how many superpixels segments is going to be generated
        # sigma: Float or (3,) array-like of floats, optional.
        # Width of Gaussian smoothing kernel for pre-processing for
        # each dimension of the image.
        segments_shape = np.shape(segments)
        print("Image", image_num, "segment shape: ", segments_shape)
        if save_sgmnt_rslt == True:
            np.save("D:/SLIC/segments/segmentImg%d_MaxSegment%d.npy" %(image_num, max_num_segments), segments) # Save all the segmentation results
        if Test_Mode == True:
            if np.amin(image_num_range) <= 4:
                # Show the output of SLIC (Single Linear Iterative Clustering)
                fig = plt.figure("Superpixels Picture %d -- %d segments -- %d iterations " %(image_num, max_num_segments, max_iters))
                # Diplay the image
                ax = fig.add_subplot(1, 1, 1) # Add a subplot. First step to combine the original image and the segments together
                # *args: Either a 3-digit integer or three separate integers describing the position of the subplot.
                # If the three integers are I, J, and K, the subplot is the Ith plot on a grid with J columns and K rows.
                ax.imshow(mark_boundaries(image, segments)) # = matplotlib.figure.add_subplot.imshow
        elif Test_Mode == False:
            # Show the output of SLIC (Single Linear Iterative Clustering)
            fig = plt.figure(
                "Superpixels Picture %d -- %d segments -- %d iterations " % (image_num, max_num_segments, max_iters))
            # Diplay the image
            ax = fig.add_subplot(1, 1,
                                 1)  # Add a subplot. First step to combine the original image and the segments together
            # *args: Either a 3-digit integer or three separate integers describing the position of the subplot.
            # If the three integers are I, J, and K, the subplot is the Ith plot on a grid with J columns and K rows.
            plt.axis("off")

        if save_sgmnt_rslt == True:
            if Test_Mode ==True:
                if np.amin(image_num_range) <= 4:
                    fig.savefig("D:/SLIC/segments/Superpixels Picture %d--Max %d segments--%d iterations.jpg" %(image_num, max_num_segments, max_iters))
            else:
                fig.savefig("D:/SLIC/segments/Superpixels Picture %d--Max %d segments--%d iterations.jpg" % (
                image_num, max_num_segments, max_iters))
    print("Superpixel image segmentation finished sucessfully.")
    del segments
else:
    print("Did not run super pixel segmentation.")
#==== Image segmentation portion END ====#

#==== Extract the index of the segments from the saved matrices ====#
if not_run_extract_sgmnt_index == False:
    # _ind: Stands for index
    # Create two matrices to store the row, column indices of the current cell. Expand the dimension later.
    print("Start extracting the index of the segmentation results...")
    largest_dim_row = 1  # Initialize the largest dimension record in the row direction among all segment fractions in the entire image dataset
    largest_dim_column = 1  # Initialize the largest dimension record in the column direction among all segment fractions in the entire image dataset
    largest_dim_row_all_sgmnts = np.zeros((max_num_segments, np.amax(image_num_range)+1)) # Create a matrix to store the largest segment fraction row dimension among all the images
    largest_dim_column_all_sgmnts = np.zeros((max_num_segments, np.amax(image_num_range)+1)) # Create a matrix to store the largest segment fraction column dimension among all the images
    for image_num in image_num_range: # Load one segement
        print("Loading the segmenation result of image number: %d/%d" %(image_num, np.amax(image_num_range)))
        # Load the actual fractions of segments in the current image
        # act_num_segments = np.load("D:/SLIC/segments/segmentImg%d_ActSegment%d.npy" %(image_num, act_num_segments))
        segments = np.load("D:/SLIC/segments/segmentImg%d_MaxSegment%d.npy" %(image_num, max_num_segments))
        # Read all the npy files one by one #
        segment_index_row_ind = np.shape(segments)[0] - 1 # Calculate the dimension of the current segmented image in row direction. Actual index = total - 1(1080 pixels)
        segment_index_column_ind = np.shape(segments)[1] - 1 # Calculate the dimension of the current segmented image in column direction. Actual index = total - 1(1920 pixels)
        act_num_segments = np.amax(segments) + 1 # Determine the actual fractions of segments in the current image
        # Initialize the matrices to store the segment indices in the image coor sys. The initialized matrix equals to the size of the image to cover all the extreme cases.
        # For example, one fraction of the segment is as large as the original image.
        #---- Create two one-row arrays to store the indices ----#
        sgmnt_fraction_row_index = np.zeros((((segment_index_row_ind+1)*(segment_index_column_ind+1)+1), act_num_segments)) #  Initialize the segment_fraction_row_index matrix. 1080 * 1920 + Act_num_segments (for 9999 storage) per img
        sgmnt_fraction_column_index = np.zeros((((segment_index_row_ind+1)*(segment_index_column_ind+1)+1), act_num_segments)) # Initialize the segment_fraction_column_index matrix. 1080 * 1920 * Act_num_segments (for 9999 storage) per img
        # Read all the segment npy files one by one END #

        # Check cell by cell to find each small segment(grid/cluster) with same number(class). Each segment fraction.
        for current_segment_num in range(0, act_num_segments, 1):
            print("Extracting image %d/%d, segment number: %d/%d" %(image_num, np.amax(image_num_range), current_segment_num+1, act_num_segments))
            segment_fraction_index_array_column_ind = 0  # Initialize the matrix column index to save the column index in the image coordinate sys of each segment fraction
            current_sgmnt_start_row_ind = 0  # Initialize the variable to store the first row index of the current segment fraction
            current_sgmnt_start_column_ind = 0  # Initialize the variable to store the first column index of the current segment fraction
            sgmnt_start_row_ind_indicator = 0 # Reset the start row indicator
            for row_index in range(0, segment_index_row_ind+1, 1): # row_index: Row index of the current cell in the image coor. Search the current segment in column direction first. Start from the first column in the first row
                sgmnt_start_column_ind_indicator = 0  # Reset the start column indicator
                for column_index in range(0, segment_index_column_ind+1, 1): # column_index: Column index of the current cell in the image coor. Search the current segment in column direction
                    if segments[row_index][column_index] == current_segment_num: # Check every cell to find out all the cells share the same value, which equals the current_segment_num
                        sgmnt_start_column_ind_indicator = sgmnt_start_column_ind_indicator + 1
                        sgmnt_start_row_ind_indicator = sgmnt_start_row_ind_indicator + 1
                        if sgmnt_start_column_ind_indicator == 1: # The current column is the start index only when the indicator == 1
                            current_sgmnt_start_column_ind = column_index
                        if sgmnt_start_row_ind_indicator == 1: # The current row is the start index only when the indicator == 1
                            current_sgmnt_start_row_ind = row_index
                        # Save the row index in the image coordinate sys of the current cell of the current_segment_num
                        sgmnt_fraction_row_index[segment_fraction_index_array_column_ind, current_segment_num] = row_index
                        # Save the column index in the image coordinate sys of the current cell of the current_segment_num
                        sgmnt_fraction_column_index[segment_fraction_index_array_column_ind, current_segment_num] = column_index
                        segment_fraction_index_array_column_ind = segment_fraction_index_array_column_ind + 1 # Move the storage array index by 1 in column direction

                        # #==== Try to find the largest dimentsion in row and column directions among all the segments (V.2)====# Wrong
                        # largest_dim_column = np.amax(sgmnt_fraction_column_index[:, current_segment_num])+1 - np.amin(sgmnt_fraction_column_index[:, current_segment_num])
                        # largest_dim_row = np.amax(sgmnt_fraction_row_index[:, current_segment_num])+1 - np.amin(sgmnt_fraction_row_index[:, current_segment_num])
                        # #==== Try to find the largest dimension in row and column directions among all the segments(V.2) END ====#

                        # #---- Try to find the largest dimension in row and column directions among all the segments (V.1)----# Wrong
                        # if ((column_index + 1) -  current_sgmnt_start_column_ind) > largest_dim_column: # Column index needs to add 1 to represent the physical column number (start from 1)
                        #     largest_dim_column = (column_index + 1) -  current_sgmnt_start_column_ind# Record the largest dimension in the row direction
                        # if ((row_index + 1) - current_sgmnt_start_row_ind) > largest_dim_row: # Row index needs to add 1 to represent the physical row number (start from 1)
                        #     largest_dim_row = (row_index + 1) - current_sgmnt_start_row_ind # Record the largest dimension in the column direction
                        # # ---- Try to find the largest dimension in row and column direction among all the segments END ----#

            # ==== Try to find the largest dimentsion in row and column directions among all the segments (V.3)====#
            largest_dim_row_all_sgmnts[current_segment_num, image_num] = np.amax(sgmnt_fraction_row_index[:, current_segment_num]) - np.amin(sgmnt_fraction_row_index[0:segment_fraction_index_array_column_ind, current_segment_num]) + 1
            largest_dim_column_all_sgmnts[current_segment_num, image_num] = np.amax(sgmnt_fraction_column_index[:, current_segment_num]) - np.amin(sgmnt_fraction_column_index[0:segment_fraction_index_array_column_ind, current_segment_num]) + 1
            # ==== Try to find the largest dimension in row and column directions among all the segments(V.3) END ====#

            print("Largest segment fraction dimension in column direction: %d" % largest_dim_column_all_sgmnts[current_segment_num, image_num])
            print("Largest segment fraction dimension in row direction: %d" % largest_dim_row_all_sgmnts[current_segment_num, image_num])
            sgmnt_fraction_row_index[segment_fraction_index_array_column_ind, current_segment_num] = 9999 # Use 9999 as a sign of the End of the segment fraction
            sgmnt_fraction_column_index[segment_fraction_index_array_column_ind, current_segment_num] = 9999 # Use 9999 as a sign of the End of the segment fraction
        # Save all the cell x, y indices in the image coordinate sys of the current image to npy files
        print("Saving the index matrices of the cells in each fraction of the segment...")
        # sgmnt_fraction_row_index.astype(np.int64)
        # sgmnt_fraction_column_index.astype(np.int64)
        np.save("D:/SLIC/segment_index/img%d_row_index.npy" %image_num, sgmnt_fraction_row_index)
        np.save("D:/SLIC/segment_index/img%d_column_index.npy" %image_num, sgmnt_fraction_column_index)
        np.save("D:/SLIC/segment_index/img%d_act_num_segments.npy" %image_num, act_num_segments) # Save the actual number of the segment fractions in the current image/segment
        print("Index matrices saved successfully.")
    # Find and save the largest dimension among row and column directions
    largest_dim_row = np.amax(largest_dim_row_all_sgmnts)
    largest_dim_column = np.amax(largest_dim_column_all_sgmnts)
    if largest_dim_column >= largest_dim_row:
        largest_dim_sgmnt_fraction = largest_dim_column
    elif largest_dim_row > largest_dim_column:
        largest_dim_sgmnt_fraction = largest_dim_row

    print("The largest segment fraction dimension is %d pixels." % largest_dim_sgmnt_fraction)
    np.save("D:/SLIC/segment_index/largest_dim.npy", largest_dim_sgmnt_fraction)
    print("Largest segment fraction dimension saved sucessfully.")
    del sgmnt_fraction_row_index
    del sgmnt_fraction_column_index
else:
    print("Did not run the index extraction program.")
#==== Extract the index of the segments from the saved matrices END ====#

#==== Pad all of the segment fractions to the dimension of (largest_dim_sgmnt_fraction by largest_dim_sgmnt_fraction) ====#
if not_run_pad_sgmnt_fractions == False:
    print("Start padding all of the segment fractions to the standard size...")
    print("Reading the largest segment fraction dimension file...")
    largest_dim_sgmnt_fraction = int(np.load("D:/SLIC/segment_index/largest_dim.npy"))
    # sgmnt_fraction_padded = np.zeros((largest_dim_sgmnt_fraction, largest_dim_sgmnt_fraction, act_num_segments)) # Initialize a standard segment fraction pad, whose dimension is largest_dim_sgmnt_fraction by largest_dim_sgmnt_fraction
    print("The size of the standard segment fraction matrix is %d by %d." %(largest_dim_sgmnt_fraction, largest_dim_sgmnt_fraction))
    # Load one segment index file
    total_num_img = np.amax(image_num_range) + 1 # Calculate the total number of images
    for image_num in image_num_range:
        act_num_segments = np.load("D:/SLIC/segment_index/img%d_act_num_segments.npy" % image_num) # Load the actual number of segment fractions in the current image/segment
        if Test_Mode == True:
            image = io.imread("C:/utils/Python/SLIC/%d.jpg" % image_num)
        else:
            image = io.imread("C:/utils/pics/image%04d.jpg" % image_num)  # For images in pics folder
        image = img_as_float(image)
        print("Loading the segmenation index matrices of image number: %d/%d" % (image_num, total_num_img))
        # Initialize a standard segment fraction pad, whose dimension is largest_dim_sgmnt_fraction by largest_dim_sgmnt_fraction
        sgmnt_fraction_padded = np.zeros((largest_dim_sgmnt_fraction, largest_dim_sgmnt_fraction, np.shape(image)[2],
                                          act_num_segments))
        sgmnt_fraction_row_index = np.load("D:/SLIC/segment_index/img%d_row_index.npy" % image_num) # Load the segment row index matrix
        sgmnt_fraction_column_index = np.load("D:/SLIC/segment_index/img%d_column_index.npy" % image_num) # Load the segment column index matrix
        # Read the pixel values in each segment fraction based on the segment fraction index arrays
        num_column = np.shape(sgmnt_fraction_row_index)[0] # Calculate the total number of columns in the segment fraction index matrices
        for current_segment_num in range(0, act_num_segments, 1): # Loop over all the segment fractions in the current image
            print("Padding Image %d, Segment %d" %(image_num, current_segment_num))
            # Initialize the indices of the fraction pad
            sgmnt_fraction_padded_row_ind = 0
            sgmnt_fraction_padded_column_ind = 0
            img_row_ind_memory = 0 # An intermediate variable to store the img_row_ind, which is used to check whether we need to move to the next row.
            min_current_sgmnt_fraction_column_ind = int(np.amin(
            sgmnt_fraction_column_index[0:int(np.argmax(sgmnt_fraction_column_index, axis = 0)[current_segment_num])+1, current_segment_num]))  # Find the minimum index in the current segment piece, which is used in the recovery of the shape of the segment piece
            # The minimum array does not count the zeors come after 9999.
            # Loop over all the cells in the current segment fraction. Column_index: The column index of the segment fraction row and column indices storage matrices.
            for column_index in range(0, num_column, 1):
                print("Segment: %d" %current_segment_num)
                print("Pad row index: %d, column index: %d" % (sgmnt_fraction_padded_row_ind, sgmnt_fraction_padded_column_ind))
                if sgmnt_fraction_row_index[column_index, current_segment_num] == 9999: # Check whether the current cell is the last cell of the current fraction
                    # column_index = num_column
                    # print("Reach the end cell of the segment.")
                    break # If the current cell is the last cell of the segment fraction, jump to the next fraction directly by breaking the current loop
                elif sgmnt_fraction_column_index[column_index, current_segment_num] == 9999:
                    # print("Reach the end cell of the segment. ")
                    break
                else:
                    row_increment_indicator = 0 # Initialize a sgmnt_fraction_padded_row_ind row increment indicator. When = 0, row index has been increased by 1 row.
                    if sgmnt_fraction_padded_column_ind + 1 == largest_dim_sgmnt_fraction: # If the current row in the pad is full, move to the next row. Also flip the row increment indicator
                        sgmnt_fraction_padded_column_ind = 0
                        sgmnt_fraction_padded_row_ind = sgmnt_fraction_padded_row_ind + 1
                        row_increment_indicator = 1 # Flip the indicator.
                        # print("Current row in the pad is full, move to the next row.")
                        # print("Pad row index: %d, colum index: %d" %(sgmnt_fraction_padded_row_ind, sgmnt_fraction_padded_column_ind))
                    # Read the indices of the current pixels in the current segment fraction
                    img_row_ind = sgmnt_fraction_row_index[column_index, current_segment_num] # Read the row index of the cell with respect to the image coor sys
                    if img_row_ind - img_row_ind_memory > 0: # Check whether moved to the following row. If so, restart the column index from 0.
                        img_row_ind_memory = img_row_ind
                        sgmnt_fraction_padded_column_ind = 0
                        # print("Reach the end cell of row %d in the image, move to the next row." % img_row_ind)
                        if row_increment_indicator == 0: # Row have not increased in the past lines
                            sgmnt_fraction_padded_row_ind = sgmnt_fraction_padded_row_ind + 1  # If the current cell is the last cell in the current row of the segment fraction, move pad index to the next row
                            # print("Reach the end cell of the current row in the image, move to the next row in the pad.")
                            # print("Pad row: %d" % sgmnt_fraction_padded_row_ind)
                    img_column_ind = sgmnt_fraction_column_index[column_index, current_segment_num] # Read the column index of the cell with respect to the image coordinate system
                    img_row_ind = int(img_row_ind)
                    img_column_ind = int(img_column_ind)
                    # Write the current pixel value to the segment fraction(padded)
                    # sgmnt_fraction_padded[:, current_segment_num, sgmnt_fraction_padded_row_ind, sgmnt_fraction_padded_column_ind] = image[img_row_ind, img_column_ind, :]

                    #---- Write the current pixel value to the segment fraction (padded) in correct order to recover the shape of the orginal piece----#
                    # min_current_sgmnt_piece_column_ind = np.amin(sgmnt_fraction_column_index[:, current_segment_num]) # Find the minimum index in the current segment piece.
                    sgmnt_fraction_padded_column_ind = img_column_ind - min_current_sgmnt_fraction_column_ind # This is the math. Convert the image index to pad index.
                    #---- Write the current pixel value to the segment fraction (padded) in correct order to recover the shape of the orginal piece END ----#

                    sgmnt_fraction_padded[sgmnt_fraction_padded_row_ind, sgmnt_fraction_padded_column_ind, :, current_segment_num] = image[img_row_ind, img_column_ind, :]

            if Test_Mode == True:
                plt.imshow(image)
                plt.imshow(sgmnt_fraction_padded[:, :, :, current_segment_num])
            plt.imsave("D:/SLIC/segment_fractions/Img_%d_sgmntPad_%d.jpg" %(image_num, current_segment_num), sgmnt_fraction_padded[:, :, :, current_segment_num])
            np.save("D:/SLIC/segment_fractions/Img_%d_sgmntPad_%d.npy" %(image_num, current_segment_num), sgmnt_fraction_padded[:, :, :, current_segment_num])
            print("Padded segment of Image %d Segment %d is saved successfully." %(image_num, current_segment_num))
else:
    print("Did not run segment fraction padding program.")