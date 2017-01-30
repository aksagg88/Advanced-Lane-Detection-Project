
# coding: utf-8

# In[104]:

# IMPORT relevant modules
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import cv2
import glob
import os
import logging
import math
import imageio
from moviepy.editor import VideoFileClip
from IPython.display import HTML
#imageio.plugins.ffmpeg.download()


PREV_LEFT_SLOPE = 0
PREV_RIGHT_SLOPE = 0
PREV_LEFT_POINTS = [0,0,0]
PREV_RIGHT_POINTS = [0,0,0]

def reset_vars():
    global PREV_LEFT_SLOPE 
    global PREV_RIGHT_SLOPE 
    global PREV_LEFT_POINTS 
    global PREV_RIGHT_POINTS 
    
    PREV_LEFT_SLOPE = 0
    PREV_RIGHT_SLOPE = 0
    PREV_LEFT_POINTS = [0,0,0]
    PREV_RIGHT_POINTS = [0,0,0]
    


# In[105]:

def camera_cal(folder,file,pattern_x,pattern_y,square_size):
    
    # pattern_x = 10 
    # pattern_y = 7
    # square_size = 1

    pattern_size = (pattern_x -1, pattern_y - 1) 
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = 0, 0
    img_names_undistort = []
    
    # create image list
    images = glob.glob('./'+folder+'/'+file+'*'+'.jpg')
    out_path = glob.glob('./'+folder+'/')
    out = False
    
    for fn in images:
        #print('processing %s... ' % fn, end='')
        img = cv2.imread(fn)
        if img is None:
            #print("Failed to load", fn)
            continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, pattern_size)
        if ret:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
            img_points.append(corners)
            #.reshape(-1, 2)
            obj_points.append(pattern_points)

        if out:
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)
            outfile = glog.glob(out_path+file+'_chess.png')
            cv2.imwrite(outfile, vis)
            if found:
                img_names_undistort.append(outfile)
        if not ret:
            #print('chessboard not found')
            continue

    #print('ok')

    # calculate camera distortion
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

    #print("\nRMS:", rms)
    #print("camera matrix:\n", camera_matrix)
    # print("matrix: \n", type(camera_matrix))
    #print("distortion coefficients: ", dist_coefs.ravel())
    
    return(camera_matrix,dist_coefs)
    
def undistort_1(image, camera_matrix, dist_coefs):
    image = cv2.undistort(image,camera_matrix,dist_coefs)
    return(image)

def undistort_2(image,camera_matrix, dist_coefs):
    h,w =image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
    undst = cv2.undistort(image, camera_matrix, dist_coefs, None, newcameramtx)
    x, y, w, h = roi
    undst = undst[y:y+h, x:x+w]
    return(undst)


# In[106]:
def test_camer_cal():

    # CALL CAMERA CALIBRATION FUNCTION
    camera_matrix, dist_coefs = camera_cal('camera_cal','calibration',10,7,1)


    # In[107]:

    # TEST DISTORTION CORRECTION

    distorted_image = mpimg.imread("./camera_cal/calibration2.jpg")
    corrected1 = undistort_1(distorted_image, camera_matrix, dist_coefs)
    corrected2 = undistort_2(distorted_image, camera_matrix, dist_coefs)


    f, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(12, 18))
    ax1.imshow(distorted_image)
    ax1.set_title('Original', fontsize=12)
    ax2.imshow(corrected1)
    ax2.set_title('Undistorted_1', fontsize=12)
    ax3.imshow(corrected2)
    ax3.set_title('Undistorted_2', fontsize=12)
    return()


# In[108]:

def warper(image, src, dst):

    # Compute and apply perpective transform
    img_size = (image.shape[1], image.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


# In[109]:

def illuminationHSV(image):
    rows,cols,channels = image.shape
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(image_hsv)
    #cv2.normalize(h, h, alpha=0, beta=127, norm_type=cv2.NORM_MINMAX)
    #cv2.normalize(s, s, alpha=0, beta=127, norm_type=cv2.NORM_MINMAX)
    #newImage = cv2.add(h,s)
    #cv2.normalize(newImage, newImage, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    if np.average(v) < 100:
        return s
    else:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def pre_process_canny(image):
    """
    Takes in an image and uses Canny Edge Detection 
    """
    # PARAMETERS
    imshape = image.shape
    kernel_size = 5
    sigma_x = 5
    sigma_y = 30
    low_canny_threshold = 25
    high_canny_threshold = low_canny_threshold * 3
    
    
    # GRAYSCALE
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = illuminationHSV(image)
    
    #equalize HIST
    #gray2 = cv2.equalizeHist(gray)
    
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray2 = clahe.apply(gray)
    
    # GAUSSIAN BLUR
    smooth = cv2.GaussianBlur(gray2, (kernel_size, kernel_size), sigma_x, sigma_y)
    
    # CANNY EDGES
    canny_edges = cv2.Canny(smooth, low_canny_threshold, high_canny_threshold)
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(canny_edges, kernel, iterations=1)
    
    if len(np.nonzero(canny_edges)[0]) > 30000:
        edges = cv2.Canny(smooth, 75, 180, apertureSize=3)
    '''    
    f, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 18))
    ax1.imshow(gray, cmap = 'Greys_r')
    ax1.set_title('grayscale', fontsize = 12)
    ax2.imshow(smooth, cmap = 'Greys_r')
    ax2.set_title('Gausian Blur', fontsize=12)
    '''
    
    return edges


# In[110]:

def colorfilter(frame):
    
    image = np.copy(frame)
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsl = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
    
   
    #Threshold the HSV image to get only yellow colors
    yellow_1 = cv2.inRange(hsv, (90, 50, 50), (100, 255, 255))
    yellow_2 = cv2.inRange(hsv, (20, 100, 100), (50, 255, 255))
    
    #Threshold the RBG image to get only white colors
    white_1 = cv2.inRange(image, (200,200,200), (255,255,255))
    
    sensitivity_1 = 68
    white_2 = cv2.inRange(hsv, (0,0,255-sensitivity_1), (255,20,255))

    sensitivity_2 = 60
    white_3 = cv2.inRange(hsl, (0,255-sensitivity_2,0), (255,255,sensitivity_2))
    
    mask = yellow_1 | yellow_2 |white_1 |white_2 |white_3
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    
    kernel = np.ones((5,5), np.uint8)
    img = cv2.erode(res, kernel, iterations=1)
    
    #plt.imshow(img[:,:,0], cmap = 'Greys_r')
    
    return(img[:,:,0])

# In[112]:

def roi_mask(image):

    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    imshape = image.shape
    ''' 
    y_max = imshape[0]-70
    y_min = 11*imshape[0]/18
    x_min = 0
    x_1 = 9*imshape[1]/20
    x_2 = 11*imshape[1]/20
    x_max = imshape[1]
    '''
    y_max = imshape[0]-70
    y_min = imshape[0]/10
    x_min = 0 + 80
    x_1 = 5*imshape[1]/20
    x_2 = 15*imshape[1]/20
    x_max = imshape[1] - 80
    
    
    vertices = np.array([[(x_min,y_max), (x_1, y_min), (x_2, y_min),(x_max,y_max)]], dtype=np.int32)
    #defining a blank mask to start with
    mask = np.zeros_like(image)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(imshape) > 2:
        channel_count = imshape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# In[113]:

# line detection

def line_detect(roi_frm, prob_hough= True):
    ret = 1
    h,w = roi_frm.shape   # height and width in pixels

    rho = 1             # distance resolution in pixels of the Hough grid
    theta = 1*np.pi/180 # angular resolution in radians of the Hough grid
    hough_threshold = 100      # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100 #minimum number of pixels making up a line
    max_line_gap = 15    # maximum gap in pixels between connectable line segments
    line_image = np.copy(roi_frm)*0 # creating a blank to draw lines on

    #print(line_image.shape)


    # Run Probabilistic Hough Transform to extract line segments from Binary image.
    lines = cv2.HoughLinesP(roi_frm,rho,theta,hough_threshold,min_line_length,max_line_gap)

    print('%d lines detect ...' % lines.shape[0],end='')

    y_max = h-20
    y_min = 0
    
    global PREV_LEFT_SLOPE, PREV_RIGHT_SLOPE, PREV_LEFT_POINTS, PREV_RIGHT_POINTS

    # DECLARE VARIABLES
    prev_weight = 0.9

    r_XY_arr = []
    r_Y_arr = []
    r_X_arr = []
    r_Slope_arr = []

    l_XY_arr = []
    l_Y_arr = []
    l_X_arr = []
    l_Slope_arr = []


    print(' processing ...',end='')
    # Loop for every single line detected by Hough Transform

    for line in lines:
        for x1,y1,x2,y2 in line:
            dx = x2 - x1
            dy = y2 - y1
            slope, yint = np.polyfit((x1, x2), (y1, y2), 1)
            theta = np.abs(np.arctan2((y2-y1), (x2-x1)))
            angle = theta * (180/np.pi)
            #angle = np.arctan2(np.array(y2-y1,dtype=np.float32),np.array(x2-x1,dtype=np.float32)) * (180/np.pi)
            if abs(angle)>20:         #for removing horizontal lines
                #print (x1,y1,x2,y2,angle)
                #cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),2)
                # Right lane marking -> positive slope and in right half of the frame
                if x1 > 0.4*w and x2 > 0.4*w and angle > 80:
                    print('right',x1,y1,x2,y2,angle)
                    r_Y_arr.extend([y1,y2])
                    r_X_arr.extend([x1,x2])
                    #r_XY_arr = np.append(r_XY_arr,[[x1,x2],[y1,y2]],1)
                    r_Slope_arr.append(slope)
                    cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)
                # left lane marking -> negative slope and in left half of the frame
                elif x1 < 0.6*w and x2 < 0.6*w and angle > 80:
                    print('left',x1,y1,x2,y2,angle)
                    l_Y_arr.extend([y1,y2])
                    l_X_arr.extend([x1,x2])
                    #l_XY_arr = np.append(l_XY_arr,[[x1,x2],[y1,y2]],1)
                    l_Slope_arr.append(slope)
                    cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)
    
    r_XY_arr = np.array((r_X_arr, r_Y_arr))
    l_XY_arr = np.array((l_X_arr, l_Y_arr))
    data_lanes = np.array((l_Slope_arr,r_Slope_arr,l_XY_arr,r_XY_arr))
                
    # DRAW RIGHT LANE LINE
    if r_Y_arr:
        r_ind = r_Y_arr.index(min(r_Y_arr))
        r_x1 = r_X_arr[r_ind]
        r_y1 = r_Y_arr[r_ind]
        r_slope = np.median(r_Slope_arr)
        print('R_slope=',r_slope)
        
        r_x_min = np.min(r_X_arr)
        r_x_max = np.max(r_X_arr)
        r_y_min = np.min(r_Y_arr)
        r_y_max = np.max(r_Y_arr)
        
        #cv2.rectangle(line_image, (r_y_min,r_x_min-10), (r_y_max,r_x_max+10), [255,255,255], 10)
        
        #UPDATE SLOPE
        if PREV_RIGHT_SLOPE !=0:
            r_slope = r_slope + (PREV_RIGHT_SLOPE - r_slope) * prev_weight
            
        if r_slope < 0.15: 
            r_x2 = r_x1;
        else:
            r_x2 = int(r_x1 + (y_max - r_y1) / r_slope)
            
        print('r_x2=',r_x2)

        # UPDATE REST OF THE COORDINATES
        if PREV_RIGHT_SLOPE !=0:
            r_x1 = int(r_x1 + (PREV_RIGHT_POINTS[0] - r_x1) * prev_weight)
            r_y1 = int(r_y1 + (PREV_RIGHT_POINTS[1] - r_y1) * prev_weight)
            r_x2 = int(r_x2 + (PREV_RIGHT_POINTS[2] - r_x2) * prev_weight)

        PREV_RIGHT_SLOPE = r_slope
        PREV_RIGHT_POINTS = [r_x1, r_y1, r_x2]
        print('Right',r_x1, r_y1, r_x2, y_max,r_slope)
        cv2.line(line_image, (r_x1, r_y1), (r_x2, y_max), (255,0,0), 15)
    else:
        print('No Right Lane!!!')
        ret = 0
        

    # DRAW LEFT LANE LINE
    if l_Y_arr:
        l_ind = l_Y_arr.index(min(l_Y_arr))
        #l_ind = l_X_arr.index(max(l_X_arr))
        l_x1 = l_X_arr[l_ind]
        l_y1 = l_Y_arr[l_ind]
        #l_slope = np.median(l_Slope_arr)
        l_slope = np.average(l_Slope_arr)
        print('l_slope=',l_slope)
        if PREV_LEFT_SLOPE !=0:
            l_slope = l_slope + (PREV_LEFT_SLOPE - l_slope) * prev_weight
            
        if l_slope < 0.15:
            l_x2 = l_x1
        else:
            l_x2 = int(l_x1 + (y_max - l_y1) / l_slope)
            
        print('l_x2=',l_x2)

        if PREV_LEFT_SLOPE !=0:
            l_x1 = int(l_x1 + (PREV_LEFT_POINTS[0] - l_x1) * prev_weight)
            l_y1 = int(l_y1 + (PREV_LEFT_POINTS[1] - l_y1) * prev_weight)
            l_x2 = int(l_x2 + (PREV_LEFT_POINTS[2] - l_x2) * prev_weight)

        PREV_LEFT_SLOPE = l_slope
        PREV_LEFT_POINTS = [l_x1, l_y1, l_x2]
        print('Left' ,l_x1, l_y1, l_x2, y_max,l_slope)
        cv2.line(line_image, (l_x1, l_y1), (l_x2, y_max), (255,0,0), 15)
    else:
        print('No Left Lane !!!')
        ret = 0
    
    return(ret,line_image,data_lanes)


# In[114]:
# In[115]:
# In[116]:
# In[137]:

# PIPELINE
def pipeline(image,camera_matrix, dist_coefs,test_im = True):
    print('Pipeline initiated ...')
    if test_im:
        reset_vars()
        
    
    frame = np.copy(image)
    img_size = frame.shape

    corrected_img = undistort_1(frame, camera_matrix, dist_coefs)
    print('Distortion corrected')
    
    #src =  np.float32([[260,680],[1045,680],[540,490],[750,490]])
    #dst = np.float32([[300,700],[1100,700],[300,50],[1100,50]])
    
    src =  np.float32([[0.203125*img_size[1],0.94444442*img_size[0]],
                       [0.81640625*img_size[1],0.94444442*img_size[0]],
                       [0.421875*img_size[1],0.68055558*img_size[0]],
                       [0.5859375*img_size[1],0.68055558*img_size[0]]])
    
    
    dst =  np.float32([[0.234375*img_size[1],0.97222221*img_size[0]],
                       [0.859375*img_size[1],0.97222221*img_size[0]],
                       [0.234375*img_size[1],0.06944445*img_size[0]],
                       [0.859375*img_size[1],0.06944445*img_size[0]]])
    

    top_view = warper(corrected_img,src,dst)
    canny_edges = pre_process_canny(top_view)
    print('Canny edges',canny_edges.shape)
    lane_pixels = colorfilter(top_view)
    print('Thresholding edges',lane_pixels.shape)
    
#    print('Edges Detected')
#    
#    hough_input = canny_edges +lane_pixels
#    line_image_1, data_lanes_1 = line_detect(hough_input,True)
#    print('Canny Hough lines found')
#    
#    # Draw the lines on the edge image
#    combo = cv2.addWeighted(line_image_1, 0.9, top_view, 1, 0)
#    dist_view_1 = warper(line_image_1,dst,src)
#    combo_2 = cv2.addWeighted(dist_view_1, 0.9, corrected_img, 1, 0)

    ret_1,line_image_1, data_lanes_1 = line_detect(canny_edges,True)
    print('Canny Hough lines found',ret_1)
    
    ret_2,line_image_2, data_lanes_2 = line_detect(lane_pixels,True)
    print('Color Hough lines found',ret_2)

    # Draw the lines on the edge image
    if (ret_1 == 1 and ret_2 == 1):
        combo = cv2.addWeighted(line_image_1, 0.9, top_view, 1, 0)
        combo = cv2.addWeighted(line_image_2, 0.9, combo, 1, 0)
        dist_view_1 = warper(line_image_1,dst,src)
        dist_view_2 = warper(line_image_2,dst,src)
        combo_2 = cv2.addWeighted(dist_view_1, 0.9, corrected_img, 1, 0)
        combo_2 = cv2.addWeighted(dist_view_2, 0.9, combo_2, 1, 0) 
        
    elif ret_1 == 0:
        combo = cv2.addWeighted(line_image_2, 0.9, top_view, 1, 0)
        dist_view_2 = warper(line_image_2,dst,src)
        combo_2 = cv2.addWeighted(dist_view_2, 0.9, corrected_img, 1, 0)
        
    elif ret_2 == 0:
        combo = cv2.addWeighted(line_image_1, 0.9, top_view, 1, 0)
        dist_view_1 = warper(line_image_1,dst,src)
        combo_2 = cv2.addWeighted(dist_view_1, 0.9, corrected_img, 1, 0)
    else:
        combo = top_view
        combo_2 = corrected_img
    
            
    # Draw the lines on the edge image
    
    
    print('... Pipeline complete!')

    # plt.imshow(roi_frm,cmap='Greys_r')
    
    f, ((ax1, ax2, ax3, ax4, ax5, ax6)) = plt.subplots(6, 1, figsize=(12, 18))
    ax1.imshow(frame)
    ax1.set_title('original image', fontsize = 12)
    ax2.imshow(top_view)
    ax2.set_title('IPM', fontsize=12)
    ax3.imshow(canny_edges, cmap = 'Greys_r')
    ax3.set_title('Canny Edges', fontsize=12)
    ax4.imshow(lane_pixels, cmap = 'Greys_r')
    ax4.set_title('Color Edges', fontsize=12)
    ax5.imshow(combo)
    ax5.set_title('Hough line', fontsize=12)
    ax6.imshow(combo_2)
    ax6.set_title('Hough line unwarped', fontsize=12)
    
    return(combo_2)


# In[138]:

# PARAMETERS
'''
#path = glob.glob('./'+'test_images'+'/'+'test6.jpg')

'''
cal_images_path = 'camera_cal'
test_images_path = glob.glob('test_images/test*.jpg')
test_videos_path = glob.glob('./test_videos/*.mp4')
output_image_path = 'output_images/'
output_video_path = 'output_videos/'
'''
camera_matrix, dist_coefs = camera_cal('camera_cal','calibration',10,7,1)

orig_img = mpimg.imread(path)
plt.imshow(orig_img)
#printing out some stats and plotting
img_size = orig_img.shape
print('This image is:', type(orig_img), 'with dimesions:', img_size)

'''



# In[139]:
'''
output = pipeline(orig_img,camera_matrix, dist_coefs,test_im = True)
cv2.imshow('output',output)

'''
# In[124]:

# Test Images
def test_images(test_images_path, save = False, plot = True):
    camera_matrix, dist_coefs = camera_cal('camera_cal','calibration',10,7,1)
    
    #fig, axs = plt.subplots(6,2, figsize=(12, 18), facecolor='w', edgecolor='k')
    #axs = axs.ravel()
    
    # create image list    
    for ind, fn in enumerate(test_images_path):
        print('processing %s... ' % fn, end='')
        img = mpimg.imread(fn)
        if img is None:
            print("Failed to load", fn)
            continue
        result = pipeline(img,camera_matrix, dist_coefs,test_im = True)
        print('ok',ind)
        if save:
            #plt.imshow(result)
            infile = os.path.basename(fn)
            outfile = (output_image_path+infile+'_output.png')
            cv2.imwrite(outfile,result)
        
    
        if plot:
            fig, ((axs1, axs2)) = plt.subplots(1,2, figsize=(12, 18), facecolor='w', edgecolor='k')
            axs1.imshow(img)
            axs1.set_title('Test', fontsize = 12)
            axs2.imshow(result)
            axs2.set_title('Output', fontsize = 12)
            plt.show()
        
        result = None
    return()


#test_images(test_images_path,False,True)
# In[ ]:
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

# In[ ]:

# Test Videos

def test_videos(video_file):

    camera_matrix, dist_coefs = camera_cal('camera_cal','calibration',10,7,1)    
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(7)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = None
    (h,w) = (None, None)
    zeros = None
    
    infile = os.path.basename(video_file)
    outfile = (output_video_path+infile+'_output.avi')
    cnt = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            cnt+= 1
            if writer is None:
                (h,w) = frame.shape[:2]
                writer = cv2.VideoWriter(outfile,fourcc, fps, (w,h),True)
                
            #frame = cv2.cvtColor(frame,cv2.COLOR_BAYER_BG2GRAY)
            result = pipeline(frame,camera_matrix, dist_coefs,test_im = False)
            writer.write(result)
            printProgressBar(cnt,total_frames,'Progress','Completed')
        else:
            break
    
    # Release everything if job is finished
    cap.release()
    writer.release()
    return()
    
#video_file = test_videos_path[2]
#test_videos(video_file)
# In[ ]:

test_img_path = '/Users/enterprise/Dev/Git/Lane-Detection/test_images/test3.jpg'
camera_matrix, dist_coefs = camera_cal('camera_cal','calibration',10,7,1)

img = mpimg.imread(test_img_path)
if img is None:
    print("Failed to load", fn)

result = pipeline(img,camera_matrix, dist_coefs,test_im = True)
print('ok',ind)

fig, ((axs1, axs2)) = plt.subplots(1,2, figsize=(12, 18), facecolor='w', edgecolor='k')
axs1.imshow(img)
axs1.set_title('Test', fontsize = 12)
axs2.imshow(result)
axs2.set_title('Output', fontsize = 12)
plt.show()


    

# In[ ]:

## SELECTING POINTS BASED ON SLOPES
def process_slopes(data_lanes):
    print(len(data_lanes[0]))
    l_slope_sdev = np.std(data_lanes[0])
    l_slope_median = np.median(data_lanes[0])
    l_slope_avg = np.average(data_lanes[0])
    l_slope_min = np.min(data_lanes[0])
    l_slope_max = np.max(data_lanes[0])
    l_angle_min = np.arctan(np.abs(l_slope_min)) * 180/(np.pi)
    l_angle_max = np.arctan(np.abs(l_slope_max)) * 180/(np.pi)
    print(l_slope_median,l_slope_avg,l_slope_sdev,l_slope_min,l_slope_max,l_angle_min,l_angle_max)

    l_slope_upper = l_slope_avg + l_slope_sdev
    l_slope_lower = l_slope_avg - l_slope_sdev
    l_angle_lower = np.arctan(np.abs(l_slope_lower)) * 180/(np.pi)
    l_angle_upper = np.arctan(np.abs(l_slope_upper)) * 180/(np.pi)
    print(l_slope_lower,l_slope_upper,l_angle_lower,l_angle_upper)

    cnt = 0;
    l_ind = []
    for sl in data_lanes[0]:
        if sl > l_slope_lower and sl<=l_slope_upper:
            l_ind = np.append(l_ind,cnt)
        cnt = cnt + 1
    print(len(l_ind))
    print(int(l_ind))
    #l_XYs = data_lanes[:,int(l_ind)]

    print(len(data_lanes[1]))
    r_slope_sdev = np.std(data_lanes[1])
    r_slope_median = np.median(data_lanes[1])
    r_slope_avg = np.average(data_lanes[1])
    r_slope_min = np.min(data_lanes[1])
    r_slope_max = np.max(data_lanes[1])
    r_angle_min = np.arctan(np.abs(r_slope_min)) * 180/(np.pi)
    r_angle_max = np.arctan(np.abs(r_slope_max)) * 180/(np.pi)
    print(r_slope_median,r_slope_avg,r_slope_sdev,r_slope_min,r_slope_max,r_angle_min,r_angle_max)
    
    return(l_slope_lower,l_slope_upper)


# In[ ]:

def ransac(image, edges, points):
    ''' Performs the RANSAC algorithm upon an input
        image and its corresponding pre-processed
        image, outputting an image with lines drawn
        on it and printing the amount of lines to stdout '''
    
    # Initialise variables
    rows,cols,channels = image.shape
    white = np.nonzero(edges)
    length = len(white[0])
    
    densestLine = -1
    bestInliers = 0
    lineCounter = 0
    iterations = 10000
    
    # While we have strong lines (probably road edges), enough edges and not
    # more than 4 lines (roads in UK don't have more 4)
    while bestInliers > (densestLine / 2) and lineCounter < 5 and length > 1000:
        # Build a bank of the locations of edge pixels
        white = np.nonzero(edges)
        length = len(white[0])
        
        # Initialise variables
        bestInliers = 0
        bestLine = []
        
        # Find lines, updating best if beaten
        for i in range (iterations + 1):
            # Select two random points
            rand1 = random.randrange(length)
            rand2 = random.randrange(length)
            x1,y1 = white[1][rand1],white[0][rand1]
            x2,y2 = white[1][rand2],white[0][rand2]
            # Heuristic to eliminate vertical and horizontal lines
            # keep selecting points until heuristics aren't violated
            cycles = 0
            while abs(x1 - x2) < 50 or abs(y1 - y2) < 50 or abs(x1 - x2) > 200:
                rand1 = random.randrange(length)
                rand2 = random.randrange(length)
                x1,y1 = white[1][rand1],white[0][rand1]
                x2,y2 = white[1][rand2],white[0][rand2]
                cycles += 1
                if cycles > 100000:
                    break
            # Create blank image
            lineImage = np.zeros((rows,cols), np.uint8)
            # Draw line between two points on blank image
            cv2.line(lineImage, (x1,y1), (x2,y2), (255,255,255), 1)
            # Logically AND the line image and processed image
            intersect = cv2.bitwise_and(edges, lineImage)
            # Calculate the number of inliers to the line
            inliers = cv2.countNonZero(intersect)
            # Modify best if necessary
            if inliers >= bestInliers:
                bestInliers = inliers
                bestLine = [(x1,y1), (x2,y2)]
            if cycles > 100000:
                break
        # Update global best to help know when to stop
        if bestInliers > densestLine:
            densestLine = bestInliers
        # Only draw line if it's a strong line (potential road edge)
        if bestInliers > (densestLine / 2):
            # Draw the line on the image
            cv2.line(image, bestLine[0], bestLine[1], (0,0,255), 1)
            # Prevent choosing this same line again my removing edges
            cv2.line(edges, bestLine[0], bestLine[1], (0,0,0), 75)
            lineCounter += 1
        if cycles > 100000:
            break
    return [image, lineCounter]




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:


