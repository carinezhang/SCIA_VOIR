import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('bd_projet_scia/damier/*')

for fname in images:
    print('begin:', fname)
    img = cv2.imread(fname)

    scale_percent = 30 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)

    kernel_size = 3
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    # Find the chess board corners
    
    ret, corners = cv2.findChessboardCorners(blur_gray, (8,9),None)
    ret2, corners2 = cv2.findChessboardCorners(blur_gray, (9, 8),None)

    # If found, add object points, image points (after refining them)
    if ret == True or ret2 == True:
        print('ok detected')
        #objpoints.append(objp)

        #corners2 = cv2.cornerSubPix(blur_gray,corners,(11,11),(-1,-1),criteria)
        #imgpoints.append(corners2)

        # Draw and display the corners
        #img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        #cv2.imshow('img',img)
        #cv2.waitKey(500)

#cv2.destroyAllWindows()