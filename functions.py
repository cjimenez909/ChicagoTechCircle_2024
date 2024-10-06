"""

    Functions that are used in the etching process within the 'etching.py' file.
    
    Calculate Corner Coordinates Author(s): Claudia Jimenez, Aima Qutbuddin
    Get Membrane Coordinates Author(s): Claudia Jimenez, Aima Qutbuddin
    Get Affine Transformation Author(s): Clayton DeVault, Claudia Jimenez
    Apply Affine Transformation Author(s): Claudia Jimenez
    Apply Affine Transformation for All Membranes Author(s): Claudia Jimenez
    Innermost Square Author(s): Kyle Cheek, Claudia Jimenez
    Square Detect Author(s): Claudia Jimenez, Aima Qutbuddin, Kyle Cheek, Lisette Ruano
        
    Commenting/Code Structure was implemented by Lisset Rico.
        
    Collaborator(s): Argonne National Laboratory (Nazar Delegan, Clayton DeVault), Break Through Tech (Kyle Cheek)
    Date Created: 06/26/2024

"""

import time
import cv2
import os
import random
import numpy as np


"""
    calculate_corner_coords : calculate theoretical GDS coordinates for 3 corners of chip
        (does not take outer edges of chip into account)

    Args:
        num_mem: number of membranes in one row of chip
        street: distance in microns of street width (region between membranes)
        mem_size: membrane length in microns
    Returns:
        corners: list of tuples with 3 corners' x, y coordinates 

"""
def calculate_corner_coords(num_mem, street, mem_size): 
    chip_length = (mem_size * num_mem) + (street * (num_mem - 1))
    
    upper_left_corner = (0, chip_length)
    lower_left_corner = (0, 0)
    upper_right_corner = (chip_length, chip_length)
    lower_right_corner = (chip_length, 0)

    corners = [lower_left_corner, upper_left_corner, upper_right_corner]

    return corners


"""
    get_mem_coords : calculates theoretical GDS coordinates of membrane centers 

    Args:
        num_mem: number of membranes in one row of chip
        street: distance in microns of street width (region between membranes)
        mem_size: membrane length in microns        
    Returns:
        coord_list : list of tuples w/ x,y coordinates of membrane centers

"""
def get_mem_coords(num_mem, street, mem_size):
    period = mem_size + street # distance in microns between each membrane

    coord_list = []

    start_mem = ((mem_size / 2), (mem_size / 2)) # lowest and leftmost membrane
    
    prev_mem = start_mem
    y = prev_mem[1]

    # traverse chip in snake motion, calculate x,y coordinates for all membrane centers, append to coord_list
    for i in range(num_mem): # row
        for j in range(num_mem): # column
            if (i == 0 and j == 0): # first membrane of whole chip
                coord_list.append(start_mem)
                continue
            elif (j == 0): # first membrane of each row
                x = prev_mem[0] # x coord unchanged from membrane directly below it
            elif (i % 2 == 0): # even row
                x = prev_mem[0] + period # go right 
            else: # odd row 
                x = prev_mem[0] - period # go left
            
            curr_mem = (x, y)
            prev_mem = curr_mem
            coord_list.append(curr_mem)

        y = prev_mem[1] + period # increase y coord for each new row
        
    return coord_list


"""
    get_affine_transform : create a matrix for an Affine transform
        to convert between GDS and stage/device coordinates

    Args:
        src_points: 3x2 numpy array of source (GDS) coordinates (3 points, each with x and y coords)
        dst_points: 3x2 numpy array of device coordinates (3 points, each with x and y coords)
    Returns:
        T: 2x3 numpy array, represents Affine transformation matrix
    Raises:
        AssertionError if input shape is incorrect

"""
def get_affine_transform(src_points, dst_points):

    # Make sure the input shape is correct
    assert src_points.shape == (3, 2) and dst_points.shape == (3, 2)

    # Create matrix A
    A = np.array([
        [src_points[0, 0], src_points[0, 1], 1, 0, 0, 0],
        [0, 0, 0, src_points[0, 0], src_points[0, 1], 1],
        [src_points[1, 0], src_points[1, 1], 1, 0, 0, 0],
        [0, 0, 0, src_points[1, 0], src_points[1, 1], 1],
        [src_points[2, 0], src_points[2, 1], 1, 0, 0, 0],
        [0, 0, 0, src_points[2, 0], src_points[2, 1], 1],
    ])

    # Create matrix B
    B = dst_points.flatten()
    
    # Solve the linear system A * x = B
    x = np.linalg.solve(A, B)

    # Reshape
    T = np.array([
        [x[0], x[1], x[2]],
        [x[3], x[4], x[5]]
    ])

    return T


"""
    apply_affine_transform : accepts GDS coordinates and returns equivalent device coordinates

    Args:
        T: Affine transform matrix (2x3 numpy array)
        src_point: tuple with 2 elements -> x,y coordinates
    Returns:
        dst_point: numpy array with x,y device coordinates

"""
def apply_affine_transform(T, src_point):

    gds = np.array([src_point[0], src_point[1], 1])
    gds = gds.transpose()
    dst_point = np.matmul(T, gds)
    dst_point = dst_point.transpose()

    return dst_point


"""
    apply_affine_all_mems : convert GDS coords to device coords for all membranes on chip

    Args:
        T: Affine transform matrix (2x3 numpy array)
        src_points_list: list of tuples w/ x,y coordinates of membrane centers
        num_mem: number of membranes in one row of chip
    Returns:
        dst_points: (total_mem)x2 numpy array w/ x,y device coordinates 
        
"""
def apply_affine_all_mems(T, src_points_list, num_mem):
    total_mem = num_mem * num_mem # for uniform square chip
    dst_points = np.zeros((total_mem, 2)) # allocate numpy array of size total_mem-by-2, fill w/ zeros

    # convert each source point to destination point 
    for (index, point) in enumerate(src_points_list):
        dst_point = apply_affine_transform(T, point)
        dst_points[index] = dst_point
        
    return dst_points

"""
    
    innermost_square : returns the most deeply nested square w/ minimum size (square of interest)
    
    Args:
        contours: list -> list of list of points that make up a contour (returned by findContours()) 
        hierarchy: list -> list of indices of contours passed in hierarchical order (returned by findContours()) 
        image: string -> image path of a given image
        min_size: integer -> minimum edge length of square to detect
    Returns:
        x1: integer -> x-coordinate of top left corner of square of interest
        y1: integer -> y-coordinate of top left corner of square of interest
        w1: integer -> width of square of interest
        h1: integer -> height of square of interest
        image: string -> image path of the original given image with a rectangle drawn on
    Raises:
        None.
    
"""
def innermost_square(contours, hierarchy, image:str, min_size:int):

    rects = [] # list for all rectangles detected
    
    # isolate all rectangles in contours
    for contour in range(len(contours)):
        (x,y,w,h) = cv2.boundingRect(contours[contour])
        rects.append((x,y,w,h))

    # make a list of all parent rectangles based on hierarchy, then sort from most to least deeply nested 
    # note: parents are sorted in hierarchical order but children are not sorted as particularly,
    # so here it is easier to sort and search by parents than by children
    parents_list = set([item[3] for items in hierarchy for item in items]) # TO DO: clean up
    parents_sorted_list = sorted(parents_list, reverse=True)

    # add parent candidates to list only if above minimum size 
    parent_candidates = []
    for i in parents_sorted_list:
        if max(rects[i][2],rects[i][3]) > min_size:
            parent_candidates.append(i)

    max_parent_candidate = max(parent_candidates) # most deeply nested parent

    # find children of minimum size and of most deeply nested parent
    child_list = []
    for (index, contour) in enumerate(hierarchy[0]):
        if (contour[3] == max_parent_candidate) and (min(rects[index][2],rects[index][3]) > min_size):
            child_list.append(index)

    # if there are child candidates, pick the most deeply nested child of the most deeply nested parent 
    # or pick the most deeply nested parent (could happen if its children do not meet min size)
    if len(child_list) > 0:
        max_child_candidate = max(child_list)
        deepest_sufficient_contour = max(max_parent_candidate, max_child_candidate)
    # if no child candidates, pick the most deeply nested parent 
    else:
        deepest_sufficient_contour = max_parent_candidate

    # get x,y of top left corner, width, and height of square of interest, draw rectangle
    (x1,y1,w1,h1) = rects[deepest_sufficient_contour]
    cv2.rectangle(image, (x1,y1), (x1+w1,y1+h1), (0,255,0), 2)

    return x1, y1, w1, h1, image


"""

    square_detect : detects whether there is a square in a given image

    Args:
        image: string -> path of image to be processed
    Returns:
        x: integer -> x-coordinate of top left corner of square of interest
        y: integer -> y-coordinate of top left corner of square of interest
        w: integer -> width of square of interest
        h: integer -> height of square of interest        
        detected: boolean -> true if a square is found, false otherwise (note: under construction)
        result: string -> copy of original image with detected square superimposed (also displayed on screen) (note: may change later)
    Raises:
        No errors. Assumes that all devices are operating correctly.
            
"""
def square_detect(img_path):
    image = cv2.imread(img_path)
    image_copy = image.copy()
    detected = False
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to filter out noise
    g_blur = cv2.GaussianBlur(gray,(5,5),0)
    
    # Apply Otsu's thresholding (automatically calculates a threshold value and binarizes image)
    ret3,otsu = cv2.threshold(g_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Invert the image (swap black and white)
    # Square will be detected better as a dark shape with a light outline
    image_binary = cv2.bitwise_not(otsu)

    # find contours
    (contours,hierarchy) = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print('Contours: ' , len(contours))
    
    # identify innermost square of min size (membrane) and identify corners
    min_size = 10
    x, y, w, h, image_rect = innermost_square(contours, hierarchy, image_copy, min_size)
 
    cv2.circle(image_rect, (x, y), 3 ,255, -1) # draw a dot on upper left corner
    cv2.circle(image_rect, (x+w, y+h), 3 ,255, -1) # draw a dot on lower right corner
    cv2.circle(image_rect, (x, y+h), 3 ,255, -1) # draw a dot on lower left corner
    cv2.circle(image_rect, (x+w, y), 3 ,255, -1) # draw a dot on upper right corner
    detected = True # TO DO: fix -> True only when at least one square detected

    cv2.destroyAllWindows()
    
    return x, y, w, h, detected
